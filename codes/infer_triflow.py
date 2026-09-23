"""Standalone inference entry point for the released TriFlow checkpoint.

This follows the active training code's inference path:

    TriFlow.sample -> flow_matching.ODESolver -> first-stage.to_volume

It deliberately does not construct the data module or start a Lightning Trainer,
so the release package can generate unconditional class-conditioned samples from
the two local checkpoints without the Academy cluster's dataset paths.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.flow.TriplaneFlow import TriFlow  # noqa: E402,F401
from util import instantiate_from_config  # noqa: E402


DEFAULT_CONFIG = CODE_ROOT / "configs" / "TriFlow.yaml"
DEFAULT_FLOW_CKPT = RELEASE_ROOT / "ckpts" / "TriFlow.ckpt"
DEFAULT_FIRST_STAGE_CKPT = RELEASE_ROOT / "ckpts" / "TriplaneVQVAE.ckpt"


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, sampling, device, and output options for inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--flow-ckpt", type=Path, default=DEFAULT_FLOW_CKPT)
    parser.add_argument("--first-stage-ckpt", type=Path, default=DEFAULT_FIRST_STAGE_CKPT)
    parser.add_argument("--output", type=Path, default=RELEASE_ROOT / "samples")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--label", type=int, default=1, choices=range(1, 7),
                        help="Training class id: 1..6.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--step-size", type=float, default=0.02,
                        help="ODE step size; 0.02 matches the active config.")
    parser.add_argument("--volume-res", type=int, default=256,
                        help="Output volume resolution per axis.")
    parser.add_argument("--amp", choices=("auto", "none", "bf16", "fp16"), default="auto")
    parser.add_argument("--no-nifti", action="store_true",
                        help="Only save latent .npy files; skip .nii.gz output.")
    return parser.parse_args()


def _check_files(args: argparse.Namespace) -> None:
    """Validate required files and basic sampling arguments before model loading."""
    for path in (args.config, args.flow_ckpt, args.first_stage_ckpt):
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")
    if args.batch_size < 1 or args.num_samples < 1:
        raise ValueError("--batch-size and --num-samples must be positive")
    if args.volume_res < 1:
        raise ValueError("--volume-res must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Load a Lightning checkpoint while keeping optimizer state out of memory."""
    print(f"Loading TriFlow checkpoint: {path}")
    load_kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        # mmap avoids eagerly duplicating the large checkpoint's tensor storage.
        checkpoint = torch.load(path, mmap=True, **load_kwargs)
    except (TypeError, RuntimeError):
        checkpoint = torch.load(path, **load_kwargs)

    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict
    del checkpoint
    print(f"TriFlow weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("Missing keys:", missing[:10])
    if unexpected:
        print("Unexpected keys:", unexpected[:10])


def build_model(config_path: Path, flow_ckpt: Path, first_stage_ckpt: Path) -> TriFlow:
    """Build the release model, load both checkpoints, and freeze inference weights.

    The first-stage training losses are removed from the temporary config so inference
    does not construct LPIPS/VGG or the discriminator.
    """
    config = OmegaConf.load(config_path)

    # The first-stage loss objects are training-only.  Removing them avoids
    # constructing LPIPS/VGG and the 3-D discriminator during inference; the
    # first-stage model architecture itself is unchanged.
    first_stage_params = config.model.params.first_stage_config.params
    for key in ("hybrid_loss_config", "gan_loss_config"):
        first_stage_params[key] = None
    config.model.params.first_stage_ckpt = str(first_stage_ckpt)

    model = instantiate_from_config(config.model)
    _load_checkpoint(model, flow_ckpt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if model.std_mean_normalize:
        if not torch.isfinite(model.global_mean).all() or not torch.isfinite(model.global_std).all():
            raise RuntimeError(
                "The TriFlow checkpoint does not contain finite latent mean/std buffers. "
                "Use the matching training checkpoint, which stores the latent statistics."
            )
    return model


def _autocast_context(device: torch.device, mode: str):
    """Return the requested CUDA autocast context, or a disabled context on CPU."""
    if mode == "none" or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    if mode == "bf16":
        dtype = torch.bfloat16
    elif mode == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"AMP dtype: {dtype}")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _save_sample(output_dir: Path, index: int, label: int, volume: torch.Tensor,
                 latent: torch.Tensor, save_nifti: bool) -> None:
    """Save one latent sample and optionally its reconstructed 1 mm NIfTI volume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_np = latent.detach().cpu().float().numpy()
    volume_np = volume.detach().cpu().float().numpy()
    if latent_np.ndim != 5 or volume_np.ndim != 5 or volume_np.shape[1] != 1:
        raise ValueError(
            f"Unexpected model output shapes: latent={latent_np.shape}, volume={volume_np.shape}"
        )
    latent_np = latent_np[0]
    volume_np = volume_np[0, 0]
    np.save(output_dir / f"sample_{index:04d}_label{label}_latent.npy", latent_np)
    np.save(output_dir / f"sample_{index:04d}_label{label}_volume.npy", volume_np)

    if save_nifti:
        import SimpleITK as sitk

        # Match TriFlow.test_step's axis convention.
        image = sitk.GetImageFromArray(volume_np.transpose(2, 1, 0))
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(output_dir / f"sample_{index:04d}_label{label}.nii.gz"))


def run_inference(args: argparse.Namespace) -> None:
    """Run batched TriFlow sampling and write all requested output artifacts.

    Sampling uses ``TriFlow.sample``; each generated item is saved as latent ``.npy``,
    volume ``.npy``, and optionally a 1 mm ``.nii.gz`` file.
    """
    _check_files(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    print(f"Device: {device}")
    model = build_model(args.config, args.flow_ckpt, args.first_stage_ckpt)
    model.to(device)
    model.ode_opts["step_size"] = args.step_size
    model.vol_res = args.volume_res

    remaining = args.num_samples
    sample_index = 0
    total_start = time.perf_counter()
    while remaining:
        current_batch = min(args.batch_size, remaining)
        labels = torch.full((current_batch,), args.label, dtype=torch.long, device=device)
        start = time.perf_counter()
        with torch.inference_mode(), _autocast_context(device, args.amp):
            volume, latent = model.sample(
                current_batch,
                device,
                extra={"label": labels, "hint": None, "plane": None},
                return_z=True,
            )
        elapsed = time.perf_counter() - start
        for batch_index in range(current_batch):
            _save_sample(
                args.output,
                sample_index,
                args.label,
                volume[batch_index:batch_index + 1],
                latent[batch_index:batch_index + 1],
                save_nifti=not args.no_nifti,
            )
            sample_index += 1
        remaining -= current_batch
        print(
            f"Generated {current_batch} sample(s) in {elapsed:.2f}s; "
            f"latent={tuple(latent.shape)}, volume={tuple(volume.shape)}"
        )

    print(f"Inference complete in {time.perf_counter() - total_start:.2f}s")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    run_inference(parse_args())
