import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from util import chunk_batch, instantiate_from_config, make_coord, resample_to_size
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only
import os, time
import mcubes
import trimesh
from omegaconf import OmegaConf
import omegaconf
from einops import rearrange
from skimage.measure import marching_cubes as mc
import SimpleITK as sitk

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from models.flow.ema import EMA
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import gc

@rank_zero_only
def rank_zero_print(*args):
    print(*args)
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def disable_training_module(module: nn.Module):
    module = module.eval()
    module.train = disabled_train
    for para in module.parameters():
        para.requires_grad = False
    return module
class WrappedUNetModel(ModelWrapper):
    """Adapt the Lightning model to the flow-matching ODE solver interface."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extra):
        with torch.no_grad():
            cond_key    = self.model.unet_config['params'].get('control_key')
            model       = self.model.model.model if isinstance(self.model.model, EMA) else self.model.model
            concat_cond = self.model.concat_cond
            cfg_scale   = self.model.cfg_scale
            stage       = self.model.stage
            
            if cond_key!=None and cond_key in extra.keys() and concat_cond:
                extra[cond_key] = torch.cat([x, extra[cond_key]], dim=1)
            
            t = (torch.zeros(x.shape[0], device=x.device) + t) * 1000
            
            if stage == 1 or cfg_scale == 1:
                result = model(x, t, extra=extra)
            else:
                x, t = torch.cat([x, x],dim=0), torch.cat([t, t],dim=0)
                for k in extra.keys():
                    extra[k] = torch.cat([extra[k], torch.zeros_like(extra[k])], dim=0)
                eps, eps_uc = model(x, t, extra=extra).chunk(2, 0)
                result = eps_uc + cfg_scale * (eps - eps_uc)
        return result

class TriFlow(pl.LightningModule):
    """Lightning module for conditional triplane flow matching and decoding."""

    def __init__(self, unet_config,
                 first_stage_config,
                 first_stage_ckpt,
                 vol_res=512,
                 latent_shape=[64,64,64],
                 ode_opts={'step_size': 0.001},
                 ode_method='midpoint',
                 std=None,
                 mean=None,
                 use_ema=False,
                 optm_betas=[0.9,0.95],
                 pretrained_ldm_ckpt=None,
                 sd_key='ldm_state_dict',
                 stage=1,
                 cfg_scale=1.0,
                 drop_scheme='default',
                 concat_cond=True,
                 cls_names_id=None):
        """Configure the flow model, frozen first-stage decoder, and latent scaling.

        Args:
            unet_config: Config entry for the flow or ControlNet backbone.
            first_stage_config: Config entry for the triplane decoder.
            first_stage_ckpt: Checkpoint for the frozen first-stage model.
            latent_shape: Shape of the flattened triplane latent sampled by the ODE.
            pretrained_ldm_ckpt: Stage-2 checkpoint providing UNet weights and stats.
            stage: ``1`` trains the flow model; ``2`` trains ControlNet only.
            cfg_scale: Classifier-free guidance scale used during sampling.
            concat_cond: Whether to concatenate the noisy latent with the condition.
        """
        super().__init__()
        self.unet_config = unet_config
        self.stage = stage
        self.cfg_scale = cfg_scale
        self.drop_scheme = drop_scheme
        self.concat_cond = concat_cond
        if cls_names_id is None:
            cls_names_id = {
                "CTChestAbdomen": 1,
                "CTHeadNeck": 2,
                "CTLegs": 3,
                "MRBody": 4,
                "MRBrain/T1": 5,
                "MRBrain/T2": 6,
            }
        self.cls_names_id = {name: int(class_id) for name, class_id in cls_names_id.items()}
        
        model = instantiate_from_config(unet_config)
        if self.stage==2:
            assert pretrained_ldm_ckpt!=None, 'ControlLDM Nedd Pre-trained LDM !!!'
            model.load_pretrained_ldm(pretrained_ldm_ckpt, sd_key)
            model.load_controlnet_from_unet()
            # Stage 2 trains only ControlNet. Keep the pretrained UNet in eval
            # mode and exclude all of its parameters from autograd while still
            # allowing gradients from the frozen decoder inputs to reach ControlNet.
            disable_training_module(model.unet)
        self.model = EMA(model=model) if use_ema else model
        
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.ode_opts = ode_opts
        self.ode_method = ode_method
        self.latent_shape = latent_shape
        
        self.first_stage_config = first_stage_config
        self.triplane = instantiate_from_config(first_stage_config)
        self.triplane.init_from_ckpt(first_stage_ckpt)
        disable_training_module(self.triplane)
        
        self.vol_res = vol_res
        self.std_mean_normalize = True
        self.register_buffer('global_mean', torch.tensor(torch.inf))
        self.register_buffer('global_std', torch.tensor(torch.inf))
        if self.stage == 2 and pretrained_ldm_ckpt is not None:
            self._load_latent_stats_from_ckpt(pretrained_ldm_ckpt)
        if mean is not None:
            self.global_mean = torch.as_tensor(mean, dtype=torch.float32)
        if std is not None:
            self.global_std = torch.as_tensor(std, dtype=torch.float32)
        
        self.optm_betas = optm_betas

    def _load_latent_stats_from_ckpt(self, ckpt_path):
        """Load latent normalization statistics from a pretrained Lightning checkpoint."""
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Pretrained LDM checkpoint does not exist: {ckpt_path}"
            )

        try:
            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            try:
                # Compatibility with PyTorch versions before mmap support.
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                # Compatibility with PyTorch versions before weights_only support.
                checkpoint = torch.load(ckpt_path, map_location="cpu")

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise KeyError(
                f"Pretrained LDM checkpoint has no state_dict: {ckpt_path}"
            )

        missing = [name for name in ("global_mean", "global_std") if name not in state_dict]
        if missing:
            raise KeyError(
                f"Pretrained LDM checkpoint is missing latent statistics {missing}: {ckpt_path}"
            )

        self.global_mean = torch.as_tensor(
            state_dict["global_mean"], dtype=torch.float32
        ).detach().clone()
        self.global_std = torch.as_tensor(
            state_dict["global_std"], dtype=torch.float32
        ).detach().clone()
        rank_zero_print(
            "Loaded latent normalization statistics from pretrained LDM: "
            f"global_mean={self.global_mean.float().mean().item():.6f}, "
            f"global_std={self.global_std.float().mean().item():.6f}"
        )
    
    def calculate_latent_stats(self, train_loader):
        total_sum = None
        total_sq_sum = None
        total_count = torch.tensor(0.0, device=self.device)
        total_min = torch.tensor(torch.inf, device=self.device)
        total_max = torch.tensor(-torch.inf, device=self.device)
    
        eps = 1e-6
        std_min = 1e-4
    
        with torch.no_grad():
            for batch in train_loader:
                latents = batch["latent"].to(self.device)
    
                if latents.ndim != 5:
                    raise ValueError(
                        f"Expected latent shape [B, 3, C, H, W], got {tuple(latents.shape)}."
                    )
                if latents.shape[1] != 3:
                    raise ValueError(
                        f"Expected 3 triplane planes at dim=1, got {latents.shape[1]}."
                    )
    
                # Use float32 for stable accumulation even if latents are fp16/bf16.
                latents = latents.float()
    
                # Sum over batch and spatial dimensions, keep [3, C].
                batch_sum = latents.sum(dim=(0, 3, 4))
                batch_sq_sum = (latents ** 2).sum(dim=(0, 3, 4))
                batch_count = latents.shape[0] * latents.shape[3] * latents.shape[4]
    
                if total_sum is None:
                    total_sum = torch.zeros_like(batch_sum, device=self.device)
                    total_sq_sum = torch.zeros_like(batch_sq_sum, device=self.device)
    
                total_sum += batch_sum
                total_sq_sum += batch_sq_sum
                total_count += batch_count
    
                total_min = torch.minimum(total_min, latents.min())
                total_max = torch.maximum(total_max, latents.max())
    
        if self.trainer.world_size > 1:
            total_sum = self.trainer.strategy.reduce(total_sum, reduce_op="sum")
            total_sq_sum = self.trainer.strategy.reduce(total_sq_sum, reduce_op="sum")
            total_count = self.trainer.strategy.reduce(total_count, reduce_op="sum")
            total_min = self.trainer.strategy.reduce(total_min, reduce_op="min")
            total_max = self.trainer.strategy.reduce(total_max, reduce_op="max")
    
        mean = total_sum / total_count
        variance = total_sq_sum / total_count - mean ** 2
        variance = torch.clamp(variance, min=0.0)
        std = torch.sqrt(variance + eps)
        std = torch.clamp(std, min=std_min)
    
        # Reshape to broadcast over [B, 3, C, H, W].
        self.global_mean = mean.view(1, mean.shape[0], mean.shape[1], 1, 1)
        self.global_std = std.view(1, std.shape[0], std.shape[1], 1, 1)
    
        rank_zero_print(
            "Calculation Done! "
            f"Plane-channel mean: mean={mean.mean().item():.6f}, "
            f"std={mean.std().item():.6f}; "
            f"Plane-channel std: mean={std.mean().item():.6f}, "
            f"min={std.min().item():.6f}, max={std.max().item():.6f}"
        )
        rank_zero_print(
            f"Latent value range: min={total_min.item():.6f}, max={total_max.item():.6f}"
        )
                
    def on_fit_start(self):
        
        from tqdm import tqdm
        
        if torch.isfinite(self.global_mean).all() and torch.isfinite(self.global_std).all():
            print(
                "Loading pre-calculated global_mean: "
                f"{self.global_mean.float().mean().item():.6f}, "
                "global_std: "
                f"{self.global_std.float().mean().item():.6f}"
            )
            return

        print("Start calculating global mean and std for 'latent'...")
        
        train_loader = self.trainer.datamodule.train_dataloader()
        
        if train_loader is None:
            raise ValueError("Train Dataloader is not accessible in on_fit_start")
        
        
        total_sum = torch.tensor(0.0, device=self.device)
        total_sq_sum = torch.tensor(0.0, device=self.device)
        total_count = torch.tensor(0.0, device=self.device)
        total_min = torch.tensor(torch.inf, device=self.device)
        total_max = torch.tensor(-torch.inf, device=self.device)

        with torch.no_grad():
            for batch in train_loader:
                latents = batch['latent']
                latents = latents.to(self.device)

                latents = latents.view(-1)
                total_sum += latents.sum()
                total_sq_sum += (latents ** 2).sum()
                total_count += latents.numel()
                total_min = min(total_min, latents.min())
                total_max = max(total_max, latents.max())

        if self.trainer.world_size > 1:
            total_sum = self.trainer.strategy.reduce(total_sum, reduce_op="sum")
            total_sq_sum = self.trainer.strategy.reduce(total_sq_sum, reduce_op="sum")
            total_count = self.trainer.strategy.reduce(total_count, reduce_op="sum")

        # Mean = Sum(x) / N
        mean = total_sum / total_count
        
        # Std = Sqrt( E[x^2] - (E[x])^2 )
        # Variance = (Sum(x^2) / N) - Mean^2
        variance = (total_sq_sum / total_count) - mean ** 2
        std = torch.sqrt(variance)

        self.global_mean = mean
        self.global_std = std
        
        rank_zero_print(f"Calculation Done! Global Mean: {mean:.6f}, Global Std: {std:.6f}")
        
        print(f"Minimum: {total_min.item():.6f}, Maximum: {total_max.item():.6f}")
        
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if isinstance(self.model, EMA):
            training   = self.model.training
            self.model.train(False)
            checkpoint["ldm_state_dict"] = self.model.model.state_dict()
            self.model.train(training)
        else:
            checkpoint["ldm_state_dict"] = self.model.state_dict()
        gc.collect()
        torch.cuda.empty_cache()
        return
        
    def normalize_latent(self, x):
        x = (x - self.global_mean) / self.global_std
        x = torch.clamp(x, min=-6.0, max=6.0)
        return x
    
    def denormalize_latent(self, x):
        x = torch.clamp(x, min=-6.0, max=6.0)
        x = x * self.global_std + self.global_mean
        return x
    
    def perpare_input(self, batch):
        """Normalize a batch and arrange latent/condition tensors for the flow model."""
        B = batch["latent"].shape[0]
        
        batch["latent"] = self.normalize_latent(batch["latent"])
        
        x = rearrange(batch["latent"], 'b n c x y -> b (n c) x y').contiguous()
        y = batch["label"].long()[:,0]
        plane = None
        
        hint = batch.get('cond')
        hint = self.normalize_latent(hint) if hint!=None else hint
        
        if self.stage==2 :
            assert hint!=None, "ControlLDM Need Condition Input !!!"
            hint = rearrange(hint, 'b n c x y -> b (n c) x y').contiguous()
        t = torch.rand((B,)).to(x.device)
        
        return x, y, t, hint, plane, B
    
    def drop(self, cond, mask):
        """Apply a per-sample conditioning mask without changing tensor shape."""
        if cond!=None:
            shape = cond.shape
            B = shape[0]
            cond = mask.view(B,*[1 for _ in range(len(shape)-1)]).to(dtype=cond.dtype) * cond
        return cond
        
    def get_drop_scheme(self, B, device):
        """Sample classifier-free guidance dropout masks for labels and conditions."""
        if self.drop_scheme=='default':
            random = torch.rand(B, dtype=torch.float32, device=device)
            drop_hint  = (random > 0.05) & (random <= 0.1)
            drop_class = (random > 0.1) & (random <= 0.15)
            drop_all = random <= 0.05
        else:
            raise NotImplementedError
        return drop_class, drop_hint, drop_all
    @torch.no_grad()
    def calc_per_class_loss(self, pred, u_t, y, log_every=50):
        """Accumulate and periodically log MSE broken down by conditioning class."""
        cls_names = {
            0: "All",
            **{class_id: name.replace('/', '_') for name, class_id in self.cls_names_id.items()},
        }
        num_classes = max(cls_names) + 1
    
        per_sample_loss = (pred - u_t).pow(2).reshape(y.shape[0], -1).mean(dim=-1).detach()
        y = y.long()
    
        if not hasattr(self, "per_class_loss_sum"):
            self.per_class_loss_sum = torch.zeros(num_classes, device=pred.device)
            self.per_class_loss_count = torch.zeros(num_classes, device=pred.device)
    
        if self.per_class_loss_sum.device != pred.device:
            self.per_class_loss_sum = self.per_class_loss_sum.to(pred.device)
            self.per_class_loss_count = self.per_class_loss_count.to(pred.device)
    
        batch_sum = torch.zeros(num_classes, device=pred.device)
        batch_count = torch.zeros(num_classes, device=pred.device)
    
        # All
        batch_sum[0] = per_sample_loss.sum()
        batch_count[0] = per_sample_loss.numel()
    
        # Known class labels from cls_names_id.
        valid = (y > 0) & (y < num_classes)
        batch_sum.scatter_add_(0, y[valid], per_sample_loss[valid])
        batch_count.scatter_add_(0, y[valid], torch.ones_like(per_sample_loss[valid]))
    
        self.per_class_loss_sum += batch_sum
        self.per_class_loss_count += batch_count
    
        if self.global_step % log_every == 0:
            global_sum = self.per_class_loss_sum.detach().clone()
            global_count = self.per_class_loss_count.detach().clone()
    
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    
            global_mean = global_sum / global_count.clamp_min(1)
    
            if self.trainer.is_global_zero:
                for i in range(num_classes):
                    if global_count[i].item() > 0:
                        self.logger.experiment.add_scalar(
                            f"{cls_names[i]}_class_loss",
                            global_mean[i].item(),
                            self.global_step,
                        )
        return                

           
    def training_step(self, batch, batch_idx):
        """Sample a flow-matching path and optimize the model's velocity prediction."""
        torch.cuda.empty_cache()
        x, y, t, hint, plane, B = self.perpare_input(batch)
        device = x.device
        
        x_1 = x
        x_0 = torch.randn_like(x_1).to(device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t, u_t = path_sample.x_t, path_sample.dx_t
        hint = torch.cat([x_t, hint], dim=1) if self.concat_cond and hint!=None else hint
        
        drop_class, drop_hint, drop_all = self.get_drop_scheme(B, device)
        y_mask    = 1.0 - (drop_class | drop_all).float()
        y         = self.drop(y, y_mask)
        hint_mask = 1.0 - (drop_hint | drop_all).float()
        hint      = self.drop(hint, hint_mask)
        
        pred = self.model(x_t, t*1000, extra={'label': y, 'hint': hint, 'plane': plane})
        loss = torch.pow(pred - u_t, 2).mean()
        
        self.calc_per_class_loss(pred, u_t, y)
        
        self.log('loss', loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=True)
        return loss
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if isinstance(self.model, EMA):
            self.model.update_ema()
        elif (
            isinstance(self.model, DistributedDataParallel)
            and isinstance(self.model.module, EMA)
        ):
            self.model.module.update_ema()
        return
       
    def configure_optimizers(self):
        """Optimize all flow weights in stage 1 or ControlNet weights in stage 2."""
        if isinstance(self.model, EMA):
            params = self.model.model.parameters() if self.stage==1 else self.model.model.controlnet.parameters()
        else:
            params = self.model.parameters() if self.stage==1 else self.model.controlnet.parameters()
        opt = torch.optim.Adam(params, lr=self.learning_rate, betas=self.optm_betas, eps=1e-6)
        return [opt], []
        
    
    @torch.no_grad()
    def sample(self, B, device, extra, return_z=False):
        """Generate normalized triplane latents with ODE sampling and decode to volume."""
        time_grid = torch.tensor([0.0, 1.0], device=device)
        x_0 = torch.randn((B, *self.latent_shape), dtype=torch.float32, device=device)
        
        solver = ODESolver(velocity_model=WrappedUNetModel(self))
        samples = solver.sample(time_grid=time_grid,
                                x_init=x_0,
                                method=self.ode_method,
                                return_intermediates=False,
                                atol=self.ode_opts["atol"] 
                                if "atol" in self.ode_opts 
                                else 1e-5,
                                rtol=self.ode_opts["rtol"] 
                                if "atol" in self.ode_opts 
                                else 1e-5,
                                step_size=self.ode_opts["step_size"]
                                if "step_size" in self.ode_opts
                                else None,
                                **extra) 
        
        samples = rearrange(samples, 'b (n c) x y -> b n c x y', n=3).contiguous()
        
        samples = self.denormalize_latent(samples)
        with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
            volume = self.triplane.to_volume(samples, (self.vol_res,)*3, quantize=False)
        
        if return_z:
            return volume, samples
        else:
            return volume
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Evaluate flow loss and save a small validation reconstruction sample."""
        torch.cuda.empty_cache()
        x, y, t, hint, plane, B = self.perpare_input(batch)
        device = x.device
        
        x_1 = x
        x_0 = torch.randn_like(x_1).to(device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t, u_t = path_sample.x_t, path_sample.dx_t
        
        pred = self.model(x_t, t*1000, extra={'label': y, 'hint': torch.cat([x_t, hint], dim=1) if self.concat_cond and hint!=None else hint, 'plane': plane})
        loss = torch.pow(pred - u_t, 2).mean()
        
        self.log('val/loss', loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        if batch_idx>=1:
            return
        
        out_dir = os.path.join(self.val_dir, f"step-{self.global_step:06d}")
        os.makedirs(out_dir, exist_ok=True)
        
        if self.stage==1:
            if self.trainer.is_global_zero:
                cls_name = {class_id: name for name, class_id in self.cls_names_id.items()}
                num = 5
                for cls, name in sorted(cls_name.items()):
                    for i in range(num):
                        t1 = time.time()
                        volume, z = self.sample(1, device, extra={'label': torch.full((1,), cls).to(y), 'hint': None, 'plane': torch.tensor([[0,1,2]]).to(plane)}, return_z=True)
                        volume, z = volume[0,0].detach().cpu().float().numpy(), z[0].detach().cpu().float().numpy()
                        
                        out_path = os.path.join(out_dir, name, f'{i}.npy')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        np.save(out_path, z)
                        
                        img = sitk.GetImageFromArray(volume.transpose(2,1,0))
                        out_path = os.path.join(out_dir, name, f'{i}.nii.gz')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        sitk.WriteImage(img, out_path)
            
        else:
            volume = self.sample(B, device, extra={'label': y, 'hint': hint, 'plane': plane})
            volume = volume[:,0].detach().cpu().float().numpy()
        
            gt, spc, org, dct, name = batch['inputs'].detach().cpu().numpy()[:,0], \
                                      batch['spc'].detach().cpu().numpy(), batch['org'].detach().cpu().numpy(), batch['dct'].detach().cpu().numpy(), batch['name']
            for i in range(B):
                img = sitk.GetImageFromArray(volume[i].transpose(2,1,0))
                img.SetSpacing(spc[i].tolist())
                img.SetOrigin(org[i].tolist())
                img.SetDirection(dct[i].tolist())
                sitk.WriteImage(img, os.path.join(out_dir, f'{name[i]}.nii.gz'))
                
                gt_img = sitk.GetImageFromArray(gt[i].transpose(2,1,0))
                gt_img.CopyInformation(img)
                sitk.WriteImage(gt_img, os.path.join(out_dir, f'{name[i]}-GT.nii.gz'))
                
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Generate test volumes or unconditional class samples and save them to disk."""
        torch.cuda.empty_cache()
        x, y, t, hint, plane, B = self.perpare_input(batch)
        device, dtype = x.device, x.dtype
        
        out_dir = self.test_dir
        os.makedirs(out_dir, exist_ok=True)
        
        if self.stage==1:
            from tqdm import tqdm
            import sys
            
            t = 0
            cls_name = {class_id: name for name, class_id in self.cls_names_id.items()}
            target_cls_ids = [1] if 1 in cls_name else [min(cls_name)]
            num = 30
            with tqdm(total=num*len(target_cls_ids), desc='Inferring...') as pbar:
                for cls in target_cls_ids:
                    for i in range(num):
                        t1 = time.time()
                        volume, z = self.sample(1, device, extra={'label': (y-y[0].item()+1)*cls, 'hint': hint, 'plane': plane}, return_z=True)
                        volume, z = volume[0,0].detach().cpu().float().numpy(), z[0].detach().cpu().float().numpy()
                        
                        t2 = time.time()
                        t += t2-t1
                        
                        out_path = os.path.join(out_dir, cls_name[cls], f'{i}.npy')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        np.save(out_path, z)
                        
                        img = sitk.GetImageFromArray(volume.transpose(2,1,0))
                        out_path = os.path.join(out_dir, cls_name[cls], f'{i}.nii.gz')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        sitk.WriteImage(img, out_path)
                        
                        pbar.set_postfix({"time": f"{t2-t1:.4f}"})
                        pbar.update(1)
            
            self.log("time", t/(num*len(target_cls_ids)), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            sys.exit(0)
            
            
            
        else:
            gt, spc, org, dct, path = batch['inputs'].detach().cpu().numpy()[:,0], \
                                      batch['spc'].detach().cpu().numpy(), batch['org'].detach().cpu().numpy(), batch['dct'].detach().cpu().numpy(), batch['path']
            
            t1 = time.time()
            volume = self.sample(B, device, extra={'label': y, 'hint': hint, 'plane': plane})
            t2 = time.time()
            self.log('time', t2-t1, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            volume = volume[:,0].detach().cpu().float().numpy()
            
            for i in range(B):
                img = sitk.GetImageFromArray(volume[i].transpose(2,1,0))
                img.SetSpacing(spc[i].tolist())
                img.SetOrigin(org[i].tolist())
                img.SetDirection(dct[i].tolist())
                out_path = os.path.join(out_dir, *path[i].strip('/').split('/')[-3:])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                sitk.WriteImage(img, out_path)
        
        
class TriControlNet(pl.LightningModule):
    """Stage-2 Lightning module that trains ControlNet against frozen TriFlow weights."""

    def __init__(self, unet_config,
                 diffusion_config,
                 first_stage_config,
                 first_stage_ckpt,
                 pretrained_ldm_ckpt,
                 controlnet_ckpt=None,
                 load_controlnet_from_unet=True,
                 vol_res=512,):
        """Load the pretrained LDM, initialize ControlNet, and freeze the decoder."""
        super().__init__()
        self.unet_config = unet_config
        self.model = instantiate_from_config(unet_config)
        self.diffusion = instantiate_from_config(diffusion_config)
        
        self.model.load_pretrained_ldm(pretrained_ldm_ckpt)
        if controlnet_ckpt!=None:
            self.model.load_controlnet_from_ckpt(controlnet_ckpt)
        elif load_controlnet_from_unet:
            init_with_new_zero, init_with_scratch = self.model.load_controlnet_from_unet()
            rank_zero_print(f"strictly load controlnet weight from pretrained SD\n"
                            f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                            f"weights initialized from scratch: {init_with_scratch}")
        else:
            rank_zero_print("Training ControlNet from scratch......")
        
        self.triplane = instantiate_from_config(first_stage_config)
        self.triplane.init_from_ckpt(first_stage_ckpt)
        disable_training_module(self.triplane)
        
        self.vol_res = vol_res
        
    def training_step(self, batch, batch_idx):
        """Compute one diffusion denoising loss using the latent and condition batch."""
        torch.cuda.empty_cache()
        x = batch["latent"]
        y = batch["label"].long()
        hint = batch["cond"]
        B = x.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=x.device)
        
        if len(self.diffusion.volume_size)==3:
            hint = hint[:,None]
            x = x[:,None]
        
        loss = self.diffusion.p_losses(self.model, x, t, y=y, hint=hint)
        self.log('loss', loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        """Freeze the pretrained UNet and optimize only ControlNet parameters."""
        self.model.unet = disable_training_module(self.model.unet)
        opt = torch.optim.AdamW(list(self.model.controlnet.parameters()), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=0.001)
        scheduler = []
        return [opt], scheduler
        
    @torch.no_grad()
    def sample(self, B, device, y, hint):
        """Sample conditioned latents, decode them through the frozen first stage, and return volumes."""
        z = torch.randn(B, self.diffusion.channels, *self.diffusion.volume_size, device=device) 
        if len(self.diffusion.volume_size)==3:
            hint = hint[:,None]
        
        samples = self.diffusion.p_sample_loop(self.model, z, y=y, hint=hint)
        if samples.dim()==5:
            samples = samples[:,0]
        
        feats = self.triplane.decode(self.triplane.denormalize(samples))
        
        volume = self.triplane.to_volume(feats, (self.vol_res,)*3)
        return volume
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Log validation diffusion loss and save generated volumes with matching metadata."""
        x = batch["latent"]
        y = batch["label"].long()
        hint = batch["cond"]
        B = y.shape[0]
        device = y.device
        name = batch.get('name')
        inputs = batch.get('inputs').detach().cpu().float().numpy()
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=x.device)
        spc, org, dct = batch['spc'].detach().cpu().float(), batch['org'].detach().cpu().float(), batch['dct'].detach().cpu().float()
        
        if len(self.diffusion.volume_size)==3:
            hint = hint[:,None]
            x = x[:,None]
        
        loss = self.diffusion.p_losses(self.model, x, t, y=y, hint=hint)
        self.log('val/loss', loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        if batch_idx>=10:
            return
        
        volume = self.sample(B, device, y, hint)
        volume = volume[:,0].detach().cpu().float().numpy()
        
        out_dir = os.path.join(self.val_dir, f"step-{self.global_step:06d}")
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(B):
            gt = sitk.GetImageFromArray(inputs[i,0].transpose(2,1,0))
            gt.SetSpacing(spc[i].numpy().tolist())
            gt.SetOrigin(org[i].numpy().tolist())
            gt.SetDirection(dct[i].numpy().tolist())
            gt = resample_to_size(gt, (self.vol_res,)*3, interp=sitk.sitkLinear)
            sitk.WriteImage(gt, os.path.join(out_dir, f'{name[i]}-GT.nii.gz'))
            
            img = sitk.GetImageFromArray(volume[i].transpose(2,1,0))
            img.CopyInformation(gt)
            sitk.WriteImage(img, os.path.join(out_dir, f'{name[i]}.nii.gz'))
            
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Generate test volumes and save them using the source image metadata."""
        torch.cuda.empty_cache()
        y = batch["label"].long()
        hint = batch["cond"]
        B = y.shape[0]
        device = y.device
        name = batch.get('name')
        inputs = batch.get('inputs').detach().cpu().float().numpy()
        spc, org, dct = batch['spc'].detach().cpu().float(), batch['org'].detach().cpu().float(), batch['dct'].detach().cpu().float()

        volume = self.sample(B, device, y, hint)
        volume = volume[:,0].detach().cpu().float().numpy()
        
        out_dir = os.path.join(self.test_dir, f"step-{self.global_step:06d}")
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(B):
            gt = sitk.GetImageFromArray(inputs[i,0].transpose(2,1,0))
            gt.SetSpacing(spc[i].numpy().tolist())
            gt.SetOrigin(org[i].numpy().tolist())
            gt.SetDirection(dct[i].numpy().tolist())
            gt = resample_to_size(gt, (self.vol_res,)*3, interp=sitk.sitkLinear)
            sitk.WriteImage(gt, os.path.join(out_dir, f'{name[i]}-GT.nii.gz'))
            
            img = sitk.GetImageFromArray(volume[i].transpose(2,1,0))
            img.CopyInformation(gt)
            sitk.WriteImage(img, os.path.join(out_dir, f'{name[i]}.nii.gz'))
                
