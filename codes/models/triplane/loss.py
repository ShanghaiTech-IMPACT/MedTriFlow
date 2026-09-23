import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from util import chunk_batch, make_coord

from modules.losses.perceptual import LPIPS

import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss, (loss_real, loss_fake)

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def least_square_d_loss(logits_real, logits_fake):
    loss_real = torch.mean((1. - logits_real)**2)
    loss_fake = torch.mean(logits_fake**2)
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss, (loss_real, loss_fake)

def least_square_g_loss(logits_fake):
    loss_fake = torch.mean((1. - logits_fake)**2)
    g_loss = 0.5 * loss_fake
    return g_loss

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), []

            

class GANLoss(nn.Module):
    def __init__(self, lambda_g=1.0, lambda_gf=0.0, lambda_d=1.0, lambda_reg=0, disc_channels=32, disc_layers=3,
                 norm_layer='nn.BatchNorm3d', disc_loss_type='hinge', adaptive_g_weight=True):
        super(GANLoss, self).__init__()
        
        self.disc = NLayerDiscriminator3D(1, disc_channels, disc_layers, norm_layer=eval(norm_layer))
        
        self.lambda_g = lambda_g
        self.lambda_gf= lambda_gf
        self.lambda_d = lambda_d
        self.lambda_reg = lambda_reg * 16
        self.reg_step = 0
        self.disc_loss_type = disc_loss_type
        self.adaptive_g_weight = adaptive_g_weight
        
        if disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss
        else:
            self.disc_loss = least_square_d_loss
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 0.5).detach() #!!! 0.02
        return d_weight
    
    def calculate_R1_regularization(self, real):
        real.requires_grad_(True)
        real_logits, _ = self.disc(real)
        
        grads = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        r1_penalty = grads.pow(2).mean()
        loss = (self.lambda_reg / 2) * r1_penalty
        return loss
        
    def forward(self, recon, gt, optm_idx, ae_loss=None, last_layer=None):
        loss_dict = {}
        ae_loss = F.l1_loss(recon, gt) if ae_loss==None else ae_loss
        dtype, device = gt.dtype, gt.device
        
        if optm_idx==0:
            logits_volume_fake, pred_volume_fake = self.disc(recon)
            if self.disc_loss_type == 'least_square':
                g_volume_loss = least_square_g_loss(logits_volume_fake)
            else:
                g_volume_loss = -torch.mean(logits_volume_fake)
            loss_dict['train/g_loss'] = g_volume_loss.detach().item()
            g_feat_loss, g_feat_weight = torch.tensor(0.0, requires_grad=True, dtype=dtype).to(device), 1.0
            if self.lambda_gf>0:
                _, pred_volume_real = self.disc(gt)
                for i in range(len(pred_volume_fake)-1):
                    g_feat_loss +=  F.l1_loss(pred_volume_fake[i], pred_volume_real[i].detach())
                g_feat_loss /= len(pred_volume_fake)-1
                loss_dict['train/g_feat_loss'] = g_feat_loss.detach().item()
            
            g_weight = self.calculate_adaptive_weight(ae_loss, g_volume_loss, last_layer) if self.adaptive_g_weight and last_layer!=None and self.lambda_g>0 else 1.0
            loss_dict['train/g_weight'] = g_weight
            
            g_loss =  self.lambda_g * g_weight * g_volume_loss + self.lambda_gf * g_feat_loss
            
            loss_dict['train/weighted_g_loss'] = g_loss.detach().item()
            
            return g_loss, loss_dict
        else:
            logits_volume_real , _ = self.disc(gt.detach())
            logits_volume_fake , _ = self.disc(recon.detach())
            
            d_loss, (loss_real, loss_fake) = self.disc_loss(logits_volume_real, logits_volume_fake)
            pred_real = (logits_volume_real.detach().view(-1) > 0).float()
            pred_fake = (logits_volume_fake.detach().view(-1) <= 0).float()
            loss_dict['train/disc_acc'] = torch.cat([pred_real, pred_fake]).mean().item()
            
            loss_dict['train/d_loss'] = d_loss.detach().item()
            loss_dict['train/logits_real'] = logits_volume_real.mean().detach().item()
            loss_dict['train/logits_fake'] = logits_volume_fake.mean().detach().item()
            d_loss = self.lambda_d*d_loss
            
            if self.lambda_reg > 0 and self.reg_step==0:
                d_reg_loss = self.calculate_R1_regularization(gt) 
                loss_dict['train/d_reg_loss'] = d_reg_loss.detach().item()
                d_loss += d_reg_loss
                self.reg_step = (self.reg_step + 1) % 16
                
            return d_loss, loss_dict
    
            
class HybridLoss(nn.Module):
    def __init__(self, lpips_start_iter=0, lambda_lpips=2.0, sample_ratio=0.1, downsample_ratio=1.0, 
                 chunk_size=262144):
        super().__init__()
        self.lpips_start_iter = lpips_start_iter
        self.lambda_lpips = lambda_lpips
        self.sample_ratio = sample_ratio
        self.downsample_ratio = downsample_ratio
        
        self.chunk_size = chunk_size
        if self.lambda_lpips:
            self.lpips = LPIPS().eval()
        
    
    def get_slice_coord(self, shape, idx, ranges, plane):
        B, N = idx.shape[0], idx.shape[-1]
        device = idx.device
        if plane == 'xy':
            H, W = shape
            grid_coords = make_coord((H, W), ranges, False).to(device, dtype=torch.float)
            x, y = torch.chunk(grid_coords[None, :, :, None].repeat(B, 1, 1, N, 1), 2, -1)
            z = idx[:, None, None, :, None].repeat(1, H, W, 1, 1)
        elif plane == 'xz':
            H, D = shape
            grid_coords = make_coord((H, D), ranges, False).to(device, dtype=torch.float)
            x, z = torch.chunk(grid_coords[None, :, None].repeat(B, 1, N, 1, 1), 2, -1)
            y = idx[:, None, :, None, None].repeat(1, H, 1, D, 1)
        else:
            W, D = shape
            grid_coords = make_coord((W, D), ranges, False).to(device, dtype=torch.float)
            y, z = torch.chunk(grid_coords[None, None].repeat(B, N, 1, 1, 1), 2, -1)
            x = idx[..., None, None, None].repeat(1, 1, W, D, 1)
    
        return torch.cat([z, y, x], dim=-1)
    
            
    def calc_lpips(self, inputs, feats, axis):
        B, C, H, W, D = inputs.shape
        h, w, d = int(self.downsample_ratio*H), int(self.downsample_ratio*W), int(self.downsample_ratio*D)
        Hs, Ws, Ds = 2/H, 2/W, 2/D
        device = inputs.device
        dtype = inputs.dtype
        offset = torch.tensor([1/H, 1/W, 1/D], dtype=dtype, device=device)
        
        if axis=='x':
            non_zeror_idx = torch.argwhere(inputs.permute(0,2,1,3,4).reshape(B, H, -1).sum(-1) > -W*D)
            x = []
            y = torch.randint(0, W - w, (1,)).item() if W-w>0 else 0
            z = torch.randint(0, D - d, (1,)).item() if D-d>0 else 0
            yz_gt = []
            num = int(H*self.sample_ratio)
            for i in range(B):
                idx = non_zeror_idx[non_zeror_idx[:,0]==i,1]
                if idx.shape[0]>0:
                    choice = torch.randint(0, idx.shape[0], (num,))
                    x.append(idx[choice])
                    yz_gt.append(inputs[i,:,idx[choice],y:y+w,z:z+d])
                else:
                    idx = torch.randint(0, H, (num,)).to(device)
                    x.append(idx)
                    yz_gt.append(inputs[i,:,idx,y:y+w,z:z+d])
                
            x = torch.stack(x).to(dtype).to(device) 
            x = Hs * x + Hs/2 - 1
            yz_gt = torch.stack(yz_gt).to(dtype).to(device).permute(0,2,1,3,4).contiguous().view(B*num, 1, w, d)
            
            ranges = ((y*Ws-1, (y+w)*Ws-1), (z*Ds-1, (z+d)*Ds-1))
            pts = self.get_slice_coord((w,d), x, ranges, 'yz').reshape(B, -1, 3).to(dtype)
            offset = offset * (torch.rand((3,), dtype=dtype, device=device)*2-1)
            offset = offset[None,None].repeat(B, pts.shape[1], 1)
            pts = pts + offset
            
            yz = chunk_batch(self.query, self.chunk_size, pts, feats=feats).view(B, 1, num, w, d).permute(0,2,1,3,4).contiguous().view(B*num, 1, w, d)
            return self.lpips(yz, yz_gt).mean()
        
        elif axis=='y':
            non_zeror_idx = torch.argwhere(inputs.permute(0,3,1,2,4).reshape(B, W, -1).sum(-1) > -H*D)
            x = torch.randint(0, H - h, (1,)).item() if H-h>0 else 0
            y = []
            z = torch.randint(0, D - d, (1,)).item() if D-d>0 else 0
            xz_gt = []
            num = int(W*self.sample_ratio)
            for i in range(B):
                idx = non_zeror_idx[non_zeror_idx[:,0]==i,1]
                if idx.shape[0]>0:
                    choice = torch.randint(0, idx.shape[0], (num,))
                    y.append(idx[choice])
                    xz_gt.append(inputs[i,:,x:x+h,idx[choice],z:z+d])
                else:
                    idx = torch.randint(0, W, (num,)).to(device)
                    y.append(idx)
                    xz_gt.append(inputs[i,:,x:x+h,idx,z:z+d])
            
            y = torch.stack(y).to(dtype).to(device) 
            y = Ws * y + Ws/2 - 1
            xz_gt = torch.stack(xz_gt).to(dtype).to(device).permute(0,3,1,2,4).contiguous().view(B*num, 1, h, d)
            
            ranges = ((x*Hs-1, (x+h)*Hs-1), (z*Ds-1, (z+d)*Ds-1))
            pts = self.get_slice_coord((h,d), y, ranges, 'xz').reshape(B, -1, 3).to(dtype)
            offset = offset * (torch.rand((3,), dtype=dtype, device=device)*2-1)
            offset = offset[None,None].repeat(B, pts.shape[1], 1)
            pts = pts + offset
            
            xz = chunk_batch(self.query, self.chunk_size, pts, feats=feats).view(B, 1, h, num, d).permute(0,3,1,2,4).contiguous().view(B*num, 1, h, d)
            return self.lpips(xz, xz_gt).mean()
        
        else:
            non_zeror_idx = torch.argwhere(inputs.permute(0,4,1,2,3).reshape(B, D, -1).sum(-1) > -H*W)
            x = torch.randint(0, H - h, (1,)).item() if H-h>0 else 0
            y = torch.randint(0, W - w, (1,)).item() if W-w>0 else 0
            z = []
            xy_gt = []
            num = int(D*self.sample_ratio)
            for i in range(B):
                idx = non_zeror_idx[non_zeror_idx[:,0]==i,1]
                if idx.shape[0]>0:
                    choice = torch.randint(0, idx.shape[0], (num,))
                    z.append(idx[choice])
                    xy_gt.append(inputs[i,:,x:x+h,y:y+w,idx[choice]])
                else:
                    idx = torch.randint(0, D, (num,)).to(device)
                    z.append(idx)
                    xy_gt.append(inputs[i,:,x:x+h,y:y+w,idx])
            
            z = torch.stack(z).to(dtype).to(device)
            z = Ds * z + Ds/2 - 1 
            xy_gt = torch.stack(xy_gt).to(dtype).to(device).permute(0,4,1,2,3).contiguous().view(B*num, 1, h, w)
            
            ranges = ((x*Hs-1, (x+h)*Hs-1), (y*Ws-1, (y+w)*Ws-1))
            pts = self.get_slice_coord((h,w), z, ranges, 'xy').reshape(B, -1, 3).to(dtype)
            offset = offset * (torch.rand((3,), dtype=dtype, device=device)*2-1)
            offset = offset[None,None].repeat(B, pts.shape[1], 1)
            pts = pts + offset
            
            xy = chunk_batch(self.query, self.chunk_size, pts, feats=feats).view(B, 1, h, w, num).permute(0,4,1,2,3).contiguous().view(B*num, 1, h, w)
            return self.lpips(xy, xy_gt).mean()
            
    
    def forward(self, logits, inputs, feats, split, global_step=1000000):
        loss_dict = {}
        loss = torch.tensor(0.0, requires_grad=True, dtype=inputs.dtype).to(inputs.device)
        B, C, H, W, D = inputs.shape
        device = inputs.device
        
        if self.sample_ratio > 0 and self.lambda_lpips > 0 and global_step > self.lpips_start_iter:
            if logits.dim()!=5:
                lpips_x = self.calc_lpips(inputs, feats, 'x')
                lpips_y = self.calc_lpips(inputs, feats, 'y')
                lpips_z = self.calc_lpips(inputs, feats, 'z')
                loss_dict[f'{split}/lpips'] = (lpips_x + lpips_y + lpips_z).detach().item()
                loss += self.lambda_lpips * (lpips_x + lpips_y + lpips_z)
            else:
                x_num, y_num, z_num = int(self.sample_ratio*H), int(self.sample_ratio*W), int(self.sample_ratio*D)
                x_idx, y_idx, z_idx = torch.randint(0, H, (x_num,)).to(device), \
                                      torch.randint(0, W, (y_num,)).to(device), \
                                      torch.randint(0, D, (z_num,)).to(device) 
                yz = self.lpips(logits[:,:,x_idx].permute(0,2,1,3,4).contiguous().view(B*x_num, 1, W, D), inputs[:,:,x_idx].permute(0,2,1,3,4).contiguous().view(B*x_num, 1, W, D)).mean()
                xz = self.lpips(logits[:,:,:,y_idx].permute(0,3,1,2,4).contiguous().view(B*y_num, 1, H, D), inputs[:,:,:,y_idx].permute(0,3,1,2,4).contiguous().view(B*y_num, 1, H, D)).mean()
                xy = self.lpips(logits[:,:,:,:,z_idx].permute(0,4,1,2,3).contiguous().view(B*z_num, 1, H, W), inputs[:,:,:,:,z_idx].permute(0,4,1,2,3).contiguous().view(B*z_num, 1, H, W)).mean()
                loss_dict[f'{split}/lpips'] = (yz + xz + xy).detach().item()
                loss += self.lambda_lpips * (yz + xz + xy)
            
            
        return loss, loss_dict
