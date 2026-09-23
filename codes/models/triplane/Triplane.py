from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from einops import rearrange
from util import calculate_rvq_metrics
from vector_quantize_pytorch import VectorQuantize

class Activation(nn.Module):
    """Select the activation used by the triplane projection and query blocks."""

    def __init__(self, swish=False, ch=None):
        super(Activation, self).__init__()
        
        if swish:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.PReLU(ch)

    def forward(self, x):
        return self.act(x)

class Normalize(nn.Module):
    """Construct the configured normalization layer for a spatial tensor."""

    def __init__(self, ch, dims=3, num_groups=32, norm_type='group', affine=False):
        super(Normalize, self).__init__()
        
        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups, ch, eps=1e-6, affine=affine)
        elif norm_type == 'batch':
            self.norm = eval(f"nn.BatchNorm{dims}d")(ch, eps=1e-6, affine=affine)
        elif norm_type == 'instance':
            self.norm = eval(f"nn.InstanceNorm{dims}d")(ch, eps=1e-6, affine=affine)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(ch)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.norm(x)

class PointEmbed(nn.Module):
    """Embed normalized 3-D query coordinates with fixed Fourier features."""

    def __init__(self, hidden_dim=36, dim=128):
        super(PointEmbed, self).__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))
        return embed

class Project3D(nn.Module):
    """Project a 3-D feature volume onto its three axis-aligned planes."""

    def __init__(self, in_ch, out_ch, res, proj_conv=2, num_groups=32, norm_type='group', affine=True, swish=False):
        super(Project3D, self).__init__()
        
        self.proj_conv = proj_conv
        kernel_size = [1, 1, 1]
        if proj_conv==2:
            self.proj3d = nn.ModuleList([])
            for i in range(3):
                ks = kernel_size.copy()
                ks[i] = res[i]
                self.proj3d.append(nn.Sequential(
                                                 Activation(swish=swish, ch=in_ch),
                                                 nn.Conv3d(in_ch, out_ch, tuple(ks))))
        elif proj_conv==1:
            self.proj3d = nn.ModuleList([])
            for i in range(3):
                ks = kernel_size
                self.proj3d.append(nn.Sequential(
                                                 Normalize(in_ch, dims=3, num_groups=num_groups, norm_type=norm_type, affine=affine),
                                                 Activation(swish=swish, ch=in_ch),
                                                 nn.Conv3d(in_ch, out_ch, tuple(ks))))
        else:
            self.proj3d = nn.ModuleList([nn.Identity(),]*3)
            
    def forward(self, x):
        hs = []
        for i in range(3):
            h = self.proj3d[i](x)
            if self.proj_conv==2:
                assert h.shape[i+2]==1, f"Channel of Axis {i+2} > 1"
            h = h.mean(i+2)
            hs.append(h)
        
        return hs


class QueryMLP(nn.Module):
    """Decode concatenated plane features and coordinate embeddings into occupancy values."""

    def __init__(self, in_ch, hidden_ch, out_ch, swish=False):
        super(QueryMLP, self).__init__()
        
        self.out_mlp = nn.ModuleList([nn.Sequential(
                                                    nn.Conv1d(in_ch, hidden_ch, 1, 1, 0),
                                                    Activation(swish=swish, ch=hidden_ch),
                                                    nn.Conv1d(hidden_ch, hidden_ch, 1, 1, 0),
                                                    Activation(swish=swish, ch=hidden_ch)),
                                      nn.Sequential(
                                                    nn.Conv1d(hidden_ch, hidden_ch, 1, 1, 0),
                                                    Activation(swish=swish, ch=hidden_ch),
                                                    nn.Conv1d(hidden_ch, out_ch, 1, 1, 0)),
                                      nn.Conv1d(in_ch, hidden_ch, 1, 1, 0)])       
    
    def get_last_layer(self):
        return self.out_mlp[1][-1].weight
    
    def forward(self, h):
        out = self.out_mlp[0](h) + self.out_mlp[-1](h)
        out = self.out_mlp[1](out)
        return out

from monai.networks.nets import UNet
from models.triplane.triplane_dc_ae import Encoder, EncoderConfig, Decoder, DecoderConfig


class AdaptiveEmbScale(nn.Module):
    """Track a stable feature-to-coordinate scale ratio during training."""

    def __init__(
        self,
        init_scale=1.0,
        momentum=0.95,
        min_scale=0.001,
        max_scale=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.momentum = momentum
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.eps = eps
        self.register_buffer("scale_ema", torch.tensor(float(init_scale)))

    @torch.no_grad()
    def update(self, emb, plane_feats):
        emb_rms = emb.detach().float().pow(2).mean().sqrt()
        feat_rms = plane_feats.detach().float().pow(2).mean().sqrt()
        scale = feat_rms / (emb_rms + self.eps)
        scale = scale.clamp(self.min_scale, self.max_scale)
        self.scale_ema.mul_(self.momentum).add_(scale.to(self.scale_ema.device), alpha=1 - self.momentum)

    def forward(self, emb, plane_feats):
        if self.training:
            self.update(emb, plane_feats)
        return emb * self.scale_ema.to(dtype=emb.dtype, device=emb.device)
        
  
class TriplaneVQVAE(nn.Module):
    """Encode volumes into quantized triplanes and decode queried 3-D points."""

    def __init__(self, in_ch=1, out_ch=1, base_ch=32, res=(256, 256, 256), ch_mult=[2,3,4], feat_ch=32, proj_conv=2, mlp_ch=256, PE_ch=64,
                 embed_dim=16, n_embed=8192, decay=0.9, restart_thres=0.1, commitment_weight=1.0,
                 norm_type='group', affine=True, swish=True, num_groups=16):
        super(TriplaneVQVAE, self).__init__()
        self.embed_dim, self.n_embed = embed_dim, n_embed
        self.conv_in = nn.Conv3d(in_ch, base_ch, 3, 1, 1)
        self.encoder3D = UNet(spatial_dims=3, in_channels=base_ch, out_channels=feat_ch, channels=tuple([base_ch*mult for mult in ch_mult]), strides=tuple([2,]*(len(ch_mult)-1)),
                              norm=("GROUP", {"num_groups": num_groups, "affine": affine}))
        self.project3D = Project3D(feat_ch, feat_ch, res, proj_conv, num_groups, norm_type, affine, swish)
        
        self.vq_encoder = Encoder(EncoderConfig(
                                    in_channels=feat_ch,
                                    latent_channels=embed_dim,
                                    width_list=(64, 128, 256),
                                    depth_list=(1, 2, 4),
                                    block_type=["ResBlock", "ResBlock", "ResBlock"],
                                    norm=["trms2d", "trms2d", "trms2d"],
                                    act=["silu", "silu", "silu"])
                                )
        self.vq_decoder = Decoder(DecoderConfig(
                                    out_channels=feat_ch,
                                    latent_channels=embed_dim,
                                    width_list=(64, 128, 256),
                                    depth_list=(1, 2, 4),
                                    block_type=["ResBlock", "ResBlock", "ResBlock"],
                                    norm=["trms2d", "trms2d", "trms2d"],
                                    act=["silu", "silu", "silu"])
                                )
        self.quant_conv = nn.Identity()
        self.post_quant_conv = nn.Identity()
        self.latent_dim = embed_dim

        self.codebook = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=n_embed,
            decay=decay,
            commitment_weight=commitment_weight,
            rotation_trick=False,
            threshold_ema_dead_code=restart_thres,
            kmeans_init=True,
            kmeans_iters=50,
        )
        self.PE = PointEmbed(dim=PE_ch)
        
        self.emb_scale = AdaptiveEmbScale()
        
        total_ch = 3*feat_ch + PE_ch
        self.query_mlp = QueryMLP(total_ch, mlp_ch, out_ch, swish)
        
        self.feat_ch, self.PE_ch = feat_ch, PE_ch
    def get_params(self):
        """Return trainable autoencoder/query parameters while freezing the codebook."""
        params = list(self.conv_in.parameters()) + list(self.encoder3D.parameters()) + list(self.project3D.parameters()) + \
                 list(self.PE.parameters()) + list(self.query_mlp.parameters())
        
        params += list(self.vq_encoder.parameters()) + list(self.quant_conv.parameters()) + \
                  list(self.post_quant_conv.parameters()) + list(self.vq_decoder.parameters())
        for para in self.codebook.parameters():
            para.requires_grad = False
        return params
        
    def encode(self, x, return_zq=False):
        """Encode a volume and return decoded plane features plus VQ diagnostics."""
        x = self.conv_in(x)
        
        h = self.encoder3D(x)
        
        yz, xz, xy = self.project3D(h)
        
        h = torch.stack([yz, xz, xy], dim=1)
        
        h = self.vq_encoder(rearrange(h, 'b n c x y -> (b n) c x y').contiguous())
        BN, C, H, W = h.shape
        B, N = BN//3, 3
        z = h.reshape(B,N,self.latent_dim,H,W)
        quant, indx, emb_loss = self.codebook(rearrange(z, 'b n c h w -> b (n h w) c').contiguous())
        info = calculate_rvq_metrics(indx, self.codebook.codebook_size, False)
        quant = rearrange(quant, 'b (n h w) c -> (b n) c h w', n=N, h=H, w=W).contiguous()

        h = self.vq_decoder(quant)
        h = h.reshape(B, N, -1, h.shape[-2], h.shape[-1])

        info['z_e/mean'] = z.detach().mean().item()
        info['z_e/std']  = z.detach().std().item()
        info['z_q/mean'] = quant.detach().mean().item()
        info['z_q/std']  = quant.detach().std().item()
        info['std_ratio']= info['z_e/std'] / (info['z_q/std'] + 1e-8)

        if return_zq:
            z = quant.reshape(B,N,-1,quant.shape[-2],quant.shape[-1])

        return h, (z, emb_loss, info)
        
    
    def sample(self, pts, feats, plane):
        """Sample one plane at normalized coordinates ``pts``."""
        B, _, N, _ = pts.shape
        yz, xz, xy = feats[:,0], feats[:,1], feats[:,2]
        plane_axis = {'xy': [1,2], 'xz': [0,2], 'yz': [0,1]}
        axis = plane_axis[plane]
        return F.grid_sample(eval(plane), pts[:, :, :, axis], padding_mode="border", align_corners=False).view(B, -1, N)
    
    @torch.no_grad()
    def monitor_sampled_point_scales(
        self,
        emb: torch.Tensor,
        plane_feats: torch.Tensor,
    ) -> dict:
        """
        emb: [B, 64, N]
        plane_feats: [B, 192, N]
        """
    
        prefix = "sampled_pts/"
        eps = 1e-8
    
        emb_f = emb.detach().float()
        feat_f = plane_feats.detach().float()
    
        emb_std = emb_f.std(unbiased=False)
        feat_std = feat_f.std(unbiased=False)
        emb_rms = emb_f.square().mean().sqrt()
        feat_rms = feat_f.square().mean().sqrt()
    
        feat_ch_std = feat_f.std(dim=(0, 2), unbiased=False)
    
        return {
            f"{prefix}emb/std": emb_std.item(),
            f"{prefix}plane_feats/std": feat_std.item(),
            f"{prefix}plane_feats_over_emb/std_ratio": (feat_std / (emb_std + eps)).item(),
            f"{prefix}emb/rms": emb_rms.item(),
            f"{prefix}plane_feats/rms": feat_rms.item(),
            f"{prefix}plane_feats_over_emb/rms_ratio": (feat_rms / (emb_rms + eps)).item(),
            f"{prefix}plane_feats/channel_std_min": feat_ch_std.min().item(),
            f"{prefix}plane_feats/channel_std_max": feat_ch_std.max().item(),
        }
    
    @torch.no_grad()
    def monitor_query_mlp_first_layer(self, prefix="query_mlp/"):
        """
        Assumes first layer input is concat([plane_feats, emb], dim=1):
          plane_feats: [B, 192, N]
          emb: [B, 64, N]
    
        Supports first layer as nn.Conv1d or nn.Linear.
        """
        first = self.query_mlp.out_mlp[0][0]
    
        w = first.weight.detach().float().squeeze(-1)
        
        plane_dim = 3*self.feat_ch
        emb_dim   = self.PE_ch
        
        w_plane = w[:, :plane_dim]
        w_emb = w[:, plane_dim:plane_dim + emb_dim]
    
        plane_norm = w_plane.norm(dim=1).mean()
        emb_norm = w_emb.norm(dim=1).mean()
        eps = 1e-8
    
        return {
            f"{prefix}first_layer/plane_weight_norm_mean": plane_norm.item(),
            f"{prefix}first_layer/emb_weight_norm_mean": emb_norm.item(),
            f"{prefix}first_layer/plane_over_emb_weight_norm_ratio": (plane_norm / (emb_norm + eps)).item(),
        }
        
    def decode(self, pts, feats, return_feat=False, only_feat=False, log_sampled_pts_scales=None):
        """Query triplane features at points and optionally return intermediate features."""
        emb = self.PE(pts).permute(0,2,1)
        
        plane_feats = torch.cat([self.sample(pts[:, None], feats, 'xy'), 
                                 self.sample(pts[:, None], feats, 'xz'), 
                                 self.sample(pts[:, None], feats, 'yz')], dim=1)
        
        emb = self.emb_scale(emb, plane_feats)
        
        feats = torch.cat([plane_feats, emb], dim=1)
        
        if only_feat:
            return plane_feats
        
        out = self.query_mlp(feats)
        
        res = [out,]
        if return_feat:
            res += [plane_feats,]
        
        if log_sampled_pts_scales is not None:
            if log_sampled_pts_scales:
                info = self.monitor_sampled_point_scales(emb, plane_feats)
                info.update(self.monitor_query_mlp_first_layer())
                info["query_mlp/first_layer/plane_over_emb_effective_ratio"] = (
                    info["query_mlp/first_layer/plane_over_emb_weight_norm_ratio"]
                    * info["sampled_pts/plane_feats_over_emb/rms_ratio"]
                )
                info["query_mlp/emb_scale"] = self.emb_scale.scale_ema.detach().item()
                res += [info,]
            else:
                res += [{},]
            
        if len(res)==1:
            return res[0]
        else:
            return res
    
    def forward(self, p, x, return_zq=False):
        """Run volume encoding followed by point-wise triplane decoding."""
        feats, (z, emb_loss, info) = self.encode(x, return_zq)
        out = self.decode(p, feats)
        return out, feats, z, emb_loss, info
        
    def get_last_layer(self):
        return self.query_mlp.get_last_layer()
