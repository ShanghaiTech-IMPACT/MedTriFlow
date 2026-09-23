import pytorch_lightning as pl
import torch.nn.functional as F

import torch
import torch.nn as nn
from einops import rearrange
import math
import mcubes
import trimesh
import numpy as np
import os, time
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only, grad_norm
import SimpleITK as sitk
import traceback

from util import instantiate_from_config, chunk_batch, make_coord, set_requires_grad

from collections import OrderedDict
from copy import deepcopy

class TriplaneVQVAE(pl.LightningModule):
    """Lightning wrapper for triplane VQ reconstruction and optional GAN losses."""

    def __init__(self, model_config, 
                 scheduler_config=None,
                 chunk_size=262144, 
                 resolution=256, 
                 hybrid_loss_config=None,
                 gan_loss_config=None,
                 disc_start_iter=-1,
                 lr_disc=None,
                 gan_patch_size=[128, 128, 128],
                 lambda_recon=2.0,
                 save_gt=True,
                 save_latent=False,
                 save_zq=False,
                 mamual_backward=False,
                 finetune_from=None,
                 ignore_keys=list(),
                 **kwargs,):
        """Configure the first-stage model, reconstruction losses, and output options.

        Args:
            model_config: Config entry describing the triplane VQ autoencoder.
            chunk_size: Number of query points evaluated at once during decoding.
            resolution: Cubic output resolution used by validation/test reconstruction.
            hybrid_loss_config: Optional perceptual or regularization loss config.
            gan_loss_config: Optional adversarial loss config.
            finetune_from: Optional checkpoint used to initialize the autoencoder.
        """
        super(TriplaneVQVAE, self).__init__()
        self.model = instantiate_from_config(model_config)
        self.model_config = model_config
        self.lambda_recon = lambda_recon
        
        self.hybrid_loss_config = hybrid_loss_config
        if hybrid_loss_config!=None:
            self.hybrid_loss = instantiate_from_config(hybrid_loss_config)
            self.hybrid_loss.query = self.query
            self.hybrid_loss.chunk_size = chunk_size
        
        self.gan_loss_config = gan_loss_config
        if gan_loss_config!=None:
            self.gan_loss = instantiate_from_config(gan_loss_config)
        self.disc_start_iter = disc_start_iter
        self.lr_disc = lr_disc
        self.gan_patch_size = gan_patch_size
        
        self.automatic_optimization = True if not mamual_backward else False
        
        if finetune_from!=None:
            print(f"Finetuning from {finetune_from}....")
            self.init_model_from_ckpt(self.model, finetune_from, ignore_keys)
        
        self.scheduler_config = scheduler_config
        self.chunk_size = chunk_size
        self.resolution = resolution
        
        self.save_gt = save_gt
        self.save_latent = save_latent
        self.save_zq = save_zq
        
        self.accumulate_grad_batches = kwargs.get('accumulate_grad_batches', 1) 
    
    def init_model_from_ckpt(self, model, path, ignore_keys=list()):
        """Load only the nested first-stage model weights from a Lightning checkpoint."""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        params = OrderedDict()
        for k in sd.keys():
            if k.split('.')[0]!='model':
                continue
            nk = '.'.join(k.split('.')[1:])
            if isinstance(ignore_keys, list) and len(ignore_keys)>0:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                    else:
                        params[nk] = sd[k]
            else:
                params[nk] = sd[k]
        
        missing, unexpected = model.load_state_dict(params, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        """Restore this wrapper from a checkpoint while reporting key mismatches."""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
   
    def encode(self, inputs, return_zq=False):
        """Encode a volume into decoded triplane features and quantization metadata."""
        return self.model.encode(inputs, return_zq=return_zq)

        
    def query(self, p, feats):   
        """Query decoded triplane features at normalized 3-D coordinates."""
        logits = self.model.decode(p, feats)[:,0]
        return logits
    
    def to_volume(self, z, res, quantize=False):
        """Decode triplane latents and query them on a regular 3-D volume grid."""
        B,N,_,H,W = z.shape
        if quantize:
            z = rearrange(z, 'b n c h w -> b (n h w) c').contiguous()
            shape = z.shape
            z, _, _ = self.model.codebook(z.reshape(z.shape[0], -1, z.shape[-1]))
            z = rearrange(z, 'b (n h w) c -> b n c h w', n=N, h=H, w=W).contiguous()
        

        h = rearrange(z, 'b n c h w -> (b n) c h w').contiguous()
        h = self.model.vq_decoder(self.model.post_quant_conv(h))
        h = rearrange(h, '(b n) c h w -> b n c h w', n=3).contiguous()
        
        B, device, dtype = h.shape[0], h.device, h.dtype
        
        grid_coords = make_coord(res, ((-1,1),(-1,1),(-1,1)), True).to(device, dtype=dtype).flip(-1)
        grid_coords = grid_coords[None].repeat(B, 1, 1)
        
        logits = chunk_batch(self.query, self.chunk_size, grid_coords, feats=h)
        logits = logits.reshape(B, 1, *res).clip(-1,1)
        return logits
    
    def get_random_patch(self, inputs):
        """Sample a jittered coordinate patch and its ground-truth voxel values."""
        B, C, H, W, D = inputs.shape
        h, w, d = self.gan_patch_size
        Hs, Ws, Ds = 2/H, 2/W, 2/D
        device = inputs.device
        dtype = inputs.dtype
        offset = torch.tensor([1/H, 1/W, 1/D], dtype=dtype, device=device)
        
        x = torch.randint(0, H - h, (1,)).item() if H-h>0 else 0
        y = torch.randint(0, W - w, (1,)).item() if W-w>0 else 0
        z = torch.randint(0, D - d, (1,)).item() if D-d>0 else 0
        ranges = ((x*Hs-1, (x+h)*Hs-1), (y*Ws-1, (y+w)*Ws-1), (z*Ds-1, (z+d)*Ds-1))
        
        patch_gt = inputs[:,:,x:x+h,y:y+w,z:z+d]
        patch_coords = make_coord((h, w, d), ranges, True).to(device, dtype=dtype)
        patch_coords = patch_coords[None].repeat(B, 1, 1)
        
        offset = offset * (torch.rand((3,), dtype=dtype, device=device)*2-1)
        offset = offset[None,None].repeat(B, patch_coords.shape[1], 1)
        patch_coords = patch_coords + offset
        
        patch_coords = patch_coords.flip(-1)
        
        return patch_coords, patch_gt
    
    def forward(self, inputs, p, gt, optm_idx=None):
        """Compute the selected autoencoder, generator, or discriminator objective."""
        B, C, H, W, D = inputs.shape
        h, w, d = self.gan_patch_size
        device = inputs.device
        dtype = inputs.dtype
        
        dist = self.trainer.strategy if self.training and hasattr(self, 'trainer') and self.trainer.num_devices > 1 else None
        loss = torch.tensor(0.0, requires_grad=True, dtype=dtype).to(device)
        
        if optm_idx==None:
            feats, (z, qloss, info) = self.model.encode(inputs)
            logits, f1, pts_info = self.model.decode(p, feats, return_feat=True, log_sampled_pts_scales=(self.global_step%10==0))
            
            # Sampled Points' features and embedding scales
            if len(pts_info.keys())>0:
                self.log_dict(pts_info, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            # VQ loss
            loss += qloss
            self.log("train/qloss", qloss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            if info!=None:
                self.log_dict(info, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            # Reconstruction l1 loss
            recon_loss = F.l1_loss(logits[:,0], gt)
            self.log('train/recon_loss', recon_loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            loss += self.lambda_recon * recon_loss
            
            # Perceptual loss (LPIPS loss / other reg loss)
            if hasattr(self, 'hybrid_loss'):
                reg_loss, loss_dict = self.hybrid_loss(logits, inputs, feats, 'train', self.global_step)
                self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log('train/reg_loss', reg_loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
                loss += reg_loss
            
        elif optm_idx==0:
            feats, (z, qloss, info) = self.model.encode(inputs)
            
            logits, f1, pts_info = self.model.decode(p, feats, return_feat=True, log_sampled_pts_scales=(self.global_step%10==0))
            
            # Sampled Points' features and embedding scales
            if len(pts_info.keys())>0:
                self.log_dict(pts_info, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            # VQ loss
            loss += qloss
            self.log("train/qloss", qloss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            if info!=None:
                self.log_dict(info, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            # Reconstruction l1 loss
            recon_loss = F.l1_loss(logits[:,0], gt)
            self.log('train/recon_loss', recon_loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            loss += self.lambda_recon * recon_loss
            
            # Perceptual loss (LPIPS loss / other reg loss)
            if hasattr(self, 'hybrid_loss'):
                reg_loss, loss_dict = self.hybrid_loss(logits, inputs, feats, 'train', self.global_step)
                self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log('train/reg_loss', reg_loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
                loss += reg_loss
            
            patch_coords, patch_gt = self.get_random_patch(inputs)
            patch_recon = chunk_batch(self.query, self.chunk_size, patch_coords, feats=feats)
            patch_recon = patch_recon.reshape(B, 1, h, w, d)
                
            # GAN loss    
            last_layer = self.model.get_last_layer() if hasattr(self.model, 'get_last_layer') else None
            g_loss, loss_dict = self.gan_loss(patch_recon, patch_gt, optm_idx, ae_loss=loss, last_layer=last_layer)
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            loss += g_loss
        else:
            with torch.no_grad():
                feats, _ = self.encode(inputs)
                patch_coords, patch_gt = self.get_random_patch(inputs)
                patch_recon = chunk_batch(self.query, self.chunk_size, patch_coords, feats=feats)
                patch_recon = patch_recon.reshape(B, 1, h, w, d)
            
            d_loss, loss_dict = self.gan_loss(patch_recon.detach(), patch_gt.detach(), optm_idx)
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            loss += d_loss
            
        return loss

    def training_step(self, batch, batch_idx):
        """Compute one reconstruction/VQ/GAN training step for a volume batch."""
        p = batch.get('pts')
        gt = batch.get('gt')
        inputs = batch.get('inputs')
        torch.cuda.empty_cache()
        
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=True)
        
        if self.automatic_optimization:
            loss = self.forward(inputs, p, gt, None)
        else: 
            opts = self.optimizers()
            if isinstance(opts, list) and self.global_step >= self.disc_start_iter:
                optm_idx = batch_idx % len(opts)
                opt = opts[optm_idx]
            else:
                optm_idx = None
                opt = opts[0]
            
            ae_models = [self.model.get_params()]
            disc_model = [self.gan_loss.disc,]
            
            if optm_idx==1:
                set_requires_grad(disc_model, True)
                set_requires_grad(ae_models, False)
            else:
                set_requires_grad(ae_models, True)
                set_requires_grad(disc_model, False)
                
            
            loss = self.forward(inputs, p, gt, optm_idx)
            
            self.log('loss', loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            loss = loss / self.accumulate_grad_batches
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            
            is_gan_training = hasattr(self, 'gan_loss') and self.global_step >= self.disc_start_iter
            update_condition_met = False
            
            if not is_gan_training:
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    update_condition_met = True
            else:
                num_optimizer_runs = (batch_idx // len(opts)) + 1
                if num_optimizer_runs % self.accumulate_grad_batches == 0:
                    update_condition_met = True
            
            if update_condition_met:
                opt.step()
                opt.zero_grad()
            
            
        return loss

        
    def cal_metrics(self, recon, gt, spc, return_items=False):
        """Compute volume PSNR and slice-wise SSIM in normalized intensity space."""
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        spc = spc.numpy()
        recon = recon.detach().cpu().float().numpy()[:,0]
        gt = gt.detach().cpu().float().numpy()[:,0]
        recon, gt = (recon + 1) / 2, (gt + 1) / 2
        
        PSNR, SSIM = [], []
        for i in range(gt.shape[0]):
            psnr_value = psnr(gt[i], recon[i], data_range=1.0)
            PSNR.append(psnr_value)
            ssim_value = []
            for z in range(gt.shape[-1]):
                ssim_value.append(ssim(gt[i,:,:,z], recon[i,:,:,z], data_range=1.0, win_size=7))
            SSIM.append(np.mean(ssim_value))
        
        if return_items:
            return PSNR, SSIM
        else:
            return np.mean(PSNR), np.mean(SSIM)
        
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Reconstruct a validation batch, log metrics, and optionally save NIfTI files."""
        name = batch.get('name')
        path = batch.get('path')
        inputs = batch.get('inputs')
        spc, org, dct = batch['spc'].detach().cpu().float(), batch['org'].detach().cpu().float(), batch['dct'].detach().cpu().float()
        B = inputs.shape[0]
        
        grid_coords = make_coord((self.resolution,)*3, ((-1,1),(-1,1),(-1,1)), True).to(inputs.device, dtype=inputs.dtype).flip(-1)
        grid_coords = grid_coords[None].repeat(B, 1, 1)
        
        loss = 0
        feats, (_, qloss, _) = self.encode(inputs)
        
        loss += qloss
        self.log("val/qloss", qloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        logits = chunk_batch(self.query, self.chunk_size, grid_coords, feats=feats)
        logits = logits.reshape(B, 1, self.resolution, self.resolution, self.resolution).clip(-1,1)
        
        recon_loss = F.l1_loss(logits, inputs)
        self.log('val/recon_loss', recon_loss.detach().item(), prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        loss += self.lambda_recon * recon_loss
        
        if hasattr(self, 'hybrid_loss'):
            reg_loss, loss_dict = self.hybrid_loss(logits, inputs, feats, 'val')
            self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/reg_loss', reg_loss.detach().item(), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            loss += reg_loss
        
        self.log(f"val/loss", loss.detach().item(), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        psnr, ssim = self.cal_metrics(logits, inputs, spc)
        self.log(f"val/PSNR", psnr, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/SSIM", ssim, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        
        step = self.global_step
        out_dir = os.path.join(self.val_dir, f"step-{step:06d}")
        os.makedirs(out_dir, exist_ok=True)
        logits = logits[:,0].detach().cpu().float().numpy()
        
        for i in range(logits.shape[0]):
            img = sitk.GetImageFromArray(logits[i].transpose(2,1,0))
            img.SetSpacing(spc[i].numpy().tolist())
            img.SetOrigin(org[i].numpy().tolist())
            img.SetDirection(dct[i].numpy().tolist())
            
            names = '/'.join(path[i].strip('/').split('/')[-3:])
            p = os.path.join(out_dir, names)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            sitk.WriteImage(img, p)
            
        
        if self.save_gt:
            gt = inputs.detach().cpu().float().numpy()[:,0]
            for i in range(gt.shape[0]):
                img = sitk.GetImageFromArray(gt[i].transpose(2,1,0))
                img.SetSpacing(spc[i].numpy().tolist())
                img.SetOrigin(org[i].numpy().tolist())
                img.SetDirection(dct[i].numpy().tolist())
                
                names = '/'.join(path[i].strip('/').split('/')[-3:]).replace('.nii.gz', '-GT.nii.gz')
                p = os.path.join(out_dir, names)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sitk.WriteImage(img, p)
                
    

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Encode test volumes and optionally save latents, reconstructions, and targets."""
        name = batch.get('name')
        path = batch.get('path')
        aug_idx = batch.get('aug_idx')
        inputs = batch.get('inputs')
        spc, org, dct = batch['spc'].detach().cpu().float(), batch['org'].detach().cpu().float(), batch['dct'].detach().cpu().float()
        B = inputs.shape[0]


        loss = 0
        feats, (z, qloss, _) = self.encode(inputs, return_zq=self.save_zq)
        loss += qloss
        self.log("test/qloss", qloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.save_latent:
            latent = z
            for i in range(B):
                names = '/'.join(path[i].strip('/').split('/')[-3:]).replace('.nii.gz', f'-{aug_idx[i]}.npy' if aug_idx[i]>0 else '.npy')
                p = os.path.join(self.test_dir, names)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, latent[i].detach().cpu().float().numpy())
                
                
        if not self.resolution > 0:
            return
        
        grid_coords = make_coord((self.resolution,)*3, ((-1,1),(-1,1),(-1,1)), True).to(inputs.device, dtype=inputs.dtype).flip(-1)
        grid_coords = grid_coords[None].repeat(B, 1, 1)
        
        logits = chunk_batch(self.query, self.chunk_size, grid_coords, feats=feats)
        logits = logits.reshape(B, 1, self.resolution, self.resolution, self.resolution).clip(-1,1)

        logits = logits[:,0].detach().cpu().float().numpy()
        for i in range(logits.shape[0]):
            img = sitk.GetImageFromArray(logits[i].transpose(2, 1, 0))
            img.SetSpacing(spc[i].numpy().tolist())
            img.SetOrigin(org[i].numpy().tolist())
            img.SetDirection(dct[i].numpy().tolist())

            names = '/'.join(path[i].strip('/').split('/')[-3:]).replace('.nii.gz', f'-{aug_idx[i]}.nii.gz' if aug_idx[i]>0 else '.nii.gz')
            p = os.path.join(self.test_dir, names)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            sitk.WriteImage(img, p)


        if self.save_gt:
            gt = inputs.detach().cpu().float().numpy()[:, 0]
            for i in range(gt.shape[0]):
                img = sitk.GetImageFromArray(gt[i].transpose(2, 1, 0))
                img.SetSpacing(spc[i].numpy().tolist())
                img.SetOrigin(org[i].numpy().tolist())
                img.SetDirection(dct[i].numpy().tolist())

                names = '/'.join(path[i].strip('/').split('/')[-3:]).replace('.nii.gz', f'-{aug_idx[i]}-GT.nii.gz' if aug_idx[i]>0 else '-GT.nii.gz')
                p = os.path.join(self.test_dir, names)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sitk.WriteImage(img, p)
        
        
    def configure_optimizers(self):
        """Create the autoencoder optimizer and, when enabled, the discriminator optimizer."""
        print("Configuring Optimizers !!!")
        lr = self.learning_rate
        if self.lr_disc == None:
            self.lr_disc = lr
        
        params = self.model.get_params()
        
        if not hasattr(self, 'gan_loss'):
            opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-6)
    
            return [opt], []
        else:
            opt_g = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-6)
            opt_d = torch.optim.Adam(list(self.gan_loss.parameters()), lr=self.lr_disc, betas=(0.5, 0.9), eps=1e-6)
            return [opt_g, opt_d], []
