import importlib

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

def load_config(*yaml_files, cli_args=[]):
    """Load YAML files and merge command-line overrides into one resolved config.

    Args:
        *yaml_files: Configuration files, merged from left to right.
        cli_args: OmegaConf-style ``key=value`` overrides.
    """
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def instantiate_from_config(config):
    """Instantiate the object described by an OmegaConf ``target`` entry."""
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """Resolve a dotted ``module.object`` path to the referenced Python object."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def set_requires_grad(models, flag):
    """Enable or disable gradients for one model, parameter iterable, or a list."""
    if not isinstance(models, list):
        models = [models]
    for model in models:
        params = model.parameters() if hasattr(model, 'parameters') else model
        for param in params:
            param.requires_grad = flag


def sobel_3d_edge_detection(volume, device='cpu'):
    """Compute the 3-D Sobel edge magnitude for a volume tensor."""
    pad = torch.nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
    volume = pad(volume[None])

    sobel_kernel_x = torch.tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    sobel_kernel_y = sobel_kernel_x.transpose(2, 3)
    sobel_kernel_z = sobel_kernel_x.transpose(2, 4)

    grad_x = F.conv3d(volume, sobel_kernel_x, padding='valid')
    grad_y = F.conv3d(volume, sobel_kernel_y, padding='valid')
    grad_z = F.conv3d(volume, sobel_kernel_z, padding='valid')
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    return magnitude.squeeze().cpu()


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def resample_to_size(img, target_size, interp=sitk.sitkLinear):
    """Resample a SimpleITK image to the requested voxel size."""
    original_size = np.array(img.GetSize(), dtype=np.int64)
    original_spacing = np.array(img.GetSpacing(), dtype=np.float64)
    target_size = np.array(target_size, dtype=np.int64)
    physical_size = original_size * original_spacing
    new_spacing = physical_size / target_size

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(target_size.tolist())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interp)
    resampler.SetOutputPixelType(img.GetPixelID())
    return resampler.Execute(img)


def chunk_batch(func, chunk_size, pts, **kwargs):
    """Evaluate a point-query function in chunks to bound peak memory usage."""
    batch_points = pts.shape[1]
    out = []
    for i in range(0, batch_points, chunk_size):
        if kwargs.get('gp') is not None:
            out.append(func(
                pts[:, i:i + chunk_size],
                feats=kwargs['feats'],
                gp=kwargs['gp'][:, i:i + chunk_size],
            ))
        else:
            out.append(func(pts[:, i:i + chunk_size], feats=kwargs['feats']))
    return torch.concat(out, dim=-1)


def calculate_rvq_metrics(indices, codebook_size=1024, residual=True):
    """Return codebook usage and perplexity statistics for quantizer indices."""
    if residual:
        _, _, num_layers = indices.shape
        flat_indices = indices.reshape(-1, num_layers)
        info = {}
        for i in range(num_layers):
            layer_indices = flat_indices[:, i]
            counts = torch.bincount(layer_indices, minlength=codebook_size).float()
            info[f'usage/{i}'] = (counts > 0).sum().item()
            probs = counts / counts.sum()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            info[f'perplexity/{i}'] = torch.exp(entropy).item()
    else:
        flat_indices = indices.flatten()
        counts = torch.bincount(flat_indices, minlength=codebook_size).float()
        info = {'usage': (counts > 0).sum().item()}
        probs = counts / counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        info['perplexity'] = torch.exp(entropy).item()
    return info
