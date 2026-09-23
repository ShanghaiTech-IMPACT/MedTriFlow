from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _valid_group_num(channels: int, group_num: int) -> int:
    group_num = min(group_num, channels)
    while channels % group_num != 0 and group_num > 1:
        group_num -= 1
    return max(group_num, 1)


def build_act(act_func: Optional[str], inplace=True) -> Optional[nn.Module]:
    if act_func is None:
        return None
    name = act_func.lower()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {act_func}")


def build_norm(norm: Optional[str], num_features: Optional[int] = None, group_num: int = 32) -> Optional[nn.Module]:
    if norm is None:
        return None
    if num_features is None:
        return None
    name = norm.lower()
    if name in ("bn2d", "batch", "batchnorm"):
        return nn.BatchNorm2d(num_features)
    if name in ("gn", "group", "groupnorm", "trms2d", "rms2d", "ln2d"):
        return nn.GroupNorm(_valid_group_num(num_features, group_num), num_features)
    raise ValueError(f"Unsupported norm: {norm}")


class ConvLayer(nn.Module):
    """Minimal PyTorch version of the official EfficientViT ConvLayer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool = False,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            bias=use_bias,
        )
        self.norm = build_norm(norm, out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualBlock(nn.Module):
    """Official DCAE pattern: main branch plus optional analytic shortcut."""

    def __init__(self, main: Optional[nn.Module], shortcut: Optional[nn.Module], post_act: Optional[str] = None):
        super().__init__()
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            y = x
        elif self.shortcut is None:
            y = self.main(x)
        else:
            y = self.main(x) + self.shortcut(x)
        if self.post_act is not None:
            y = self.post_act(y)
        return y


class OpSequential(nn.Module):
    def __init__(self, op_list: Sequence[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    """Official DCAE downsample main branch: Conv -> pixel_unshuffle."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, factor: int):
        super().__init__()
        out_ratio = factor**2
        if out_channels % out_ratio != 0:
            raise ValueError(f"out_channels={out_channels} must be divisible by factor^2={out_ratio}.")
        self.factor = factor
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(self.conv(x), self.factor)


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    """Official DCAE analytic shortcut: pixel_unshuffle -> channel averaging."""

    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        if in_channels * factor**2 % out_channels != 0:
            raise ValueError(
                f"in_channels * factor^2 ({in_channels * factor**2}) must be divisible by out_channels={out_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        b, _, h, w = x.shape
        return x.view(b, self.out_channels, self.group_size, h, w).mean(dim=2)


class ConvPixelShuffleUpSampleLayer(nn.Module):
    """Official DCAE upsample main branch: Conv -> pixel_shuffle."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, factor: int):
        super().__init__()
        self.factor = factor
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * factor**2,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.conv(x), self.factor)


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    """Official DCAE analytic shortcut: channel duplication -> pixel_shuffle."""

    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        if out_channels * factor**2 % in_channels != 0:
            raise ValueError(
                f"out_channels * factor^2 ({out_channels * factor**2}) must be divisible by in_channels={in_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        return F.pixel_shuffle(x, self.factor)


class ResBlock(nn.Module):
    """Official DCAE ResBlock shape, with local GroupNorm fallback for this repo."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bias: tuple[bool, bool] = (True, False),
        norm: tuple[Optional[str], Optional[str]] = (None, "trms2d"),
        act_func: tuple[Optional[str], Optional[str]] = ("silu", None),
    ):
        super().__init__()
        self.conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class GLUMBConv(nn.Module):
    """EfficientViT local GLU block used by DCAE high-level stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expand_ratio: float = 4.0,
        norm: Optional[str] = "trms2d",
        act_func: str = "silu",
    ):
        super().__init__()
        mid_channels = round(in_channels * expand_ratio)
        self.inverted_conv = ConvLayer(in_channels, mid_channels * 2, 1, use_bias=True, norm=None, act_func=act_func)
        self.depth_conv = nn.Conv2d(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=mid_channels * 2,
            bias=True,
        )
        self.glu_act = build_act(act_func, inplace=False)
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, use_bias=False, norm=norm, act_func=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        if self.glu_act is not None:
            gate = self.glu_act(gate)
        return self.point_conv(x * gate)


class LiteMLA(nn.Module):
    """Compact copy of EfficientViT LiteMLA for low-resolution DCAE stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 32,
        heads_ratio: float = 1.0,
        scales: tuple[int, ...] = (),
        eps: float = 1.0e-15,
    ):
        super().__init__()
        heads = max(1, int(in_channels // dim * heads_ratio))
        total_dim = heads * dim
        self.dim = dim
        self.heads = heads
        self.eps = eps
        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1, use_bias=True, norm=None, act_func=None)
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3 * total_dim, 3 * total_dim, scale, padding=scale // 2, groups=3 * total_dim, bias=True),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=True),
                )
                for scale in scales
            ]
        )
        self.proj = ConvLayer(total_dim * (1 + len(scales)), out_channels, 1, use_bias=False, norm="trms2d", act_func=None)

    def _relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        b, _, h, w = qkv.shape
        qkv = qkv.reshape(b, -1, 3 * self.dim, h * w)
        q, k, v = qkv[:, :, : self.dim], qkv[:, :, self.dim : 2 * self.dim], qkv[:, :, 2 * self.dim :]
        q = F.relu(q)
        k = F.relu(k)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, k.transpose(-1, -2))
        out = torch.matmul(vk, q)
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv = torch.cat([qkv] + [op(qkv) for op in self.aggreg], dim=1)
        return self.proj(self._relu_linear_att(qkv))


class EfficientViTBlock(nn.Module):
    """DCAE's EViT_GLU/EViTS5_GLU stage block."""

    def __init__(self, in_channels: int, norm: str = "trms2d", act_func: str = "silu", scales: tuple[int, ...] = ()):
        super().__init__()
        self.context_module = ResidualBlock(LiteMLA(in_channels, in_channels, scales=scales), IdentityLayer())
        self.local_module = ResidualBlock(GLUMBConv(in_channels, in_channels, norm=norm, act_func=act_func), IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local_module(self.context_module(x))


@dataclass
class EncoderConfig:
    in_channels: int
    latent_channels: int
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: str = "trms2d"
    act: str = "silu"
    downsample_block_type: str = "ConvPixelUnshuffle"
    downsample_match_channel: bool = True
    downsample_shortcut: Optional[str] = "averaging"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None
    out_shortcut: Optional[str] = "averaging"
    double_latent: bool = False


@dataclass
class DecoderConfig:
    out_channels: int
    latent_channels: int
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: Optional[str] = "duplicating"
    out_norm: Optional[str] = "trms2d"
    out_act: Optional[str] = None


def _stage_value(value: Any, stage_id: int) -> Any:
    if isinstance(value, (list, tuple)):
        return value[stage_id]
    return value


def build_block(block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]) -> nn.Module:
    if block_type == "ResBlock":
        if in_channels != out_channels:
            main = ResBlock(in_channels, out_channels, kernel_size=3, norm=(None, norm), act_func=(act, None))
            shortcut = ConvLayer(in_channels, out_channels, 1, use_bias=True, norm=None, act_func=None)
            return ResidualBlock(main, shortcut)
        main = ResBlock(in_channels, out_channels, kernel_size=3, norm=(None, norm), act_func=(act, None))
        return ResidualBlock(main, IdentityLayer())
    if block_type == "EViT_GLU":
        if in_channels != out_channels:
            raise ValueError("EViT_GLU requires in_channels == out_channels.")
        return EfficientViTBlock(in_channels, norm=norm or "trms2d", act_func=act or "silu", scales=())
    if block_type == "EViTS5_GLU":
        if in_channels != out_channels:
            raise ValueError("EViTS5_GLU requires in_channels == out_channels.")
        return EfficientViTBlock(in_channels, norm=norm or "trms2d", act_func=act or "silu", scales=(5,))
    raise ValueError(f"Unsupported block_type: {block_type}")


def build_stage_main(
    width: int,
    depth: int,
    block_type: str | Sequence[str],
    norm: str | Sequence[str],
    act: str | Sequence[str],
    input_width: int,
) -> list[nn.Module]:
    stage: list[nn.Module] = []
    for layer_id in range(depth):
        current_block_type = block_type[layer_id] if isinstance(block_type, list) else block_type
        stage.append(
            build_block(
                current_block_type,
                in_channels=input_width if layer_id == 0 else width,
                out_channels=width,
                norm=norm if isinstance(norm, str) else norm[layer_id],
                act=act if isinstance(act, str) else act[layer_id],
            )
        )
    return stage


def build_downsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, kernel_size=3, factor=2)
    elif block_type == "Conv":
        block = ConvLayer(in_channels, out_channels, kernel_size=3, stride=2, use_bias=True, norm=None, act_func=None)
    else:
        raise ValueError(f"Unsupported downsample block type: {block_type}")
    if shortcut is None:
        return block
    if shortcut == "averaging":
        return ResidualBlock(block, PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, factor=2))
    raise ValueError(f"Unsupported downsample shortcut: {shortcut}")


def build_encoder_project_in_block(in_channels: int, out_channels: int, factor: int, downsample_block_type: str) -> nn.Module:
    if factor == 1:
        return ConvLayer(in_channels, out_channels, kernel_size=3, use_bias=True, norm=None, act_func=None)
    if factor == 2:
        return build_downsample_block(downsample_block_type, in_channels, out_channels, shortcut=None)
    raise ValueError(f"Unsupported encoder project_in factor: {factor}")


def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(in_channels, out_channels, kernel_size=3, factor=2)
    else:
        raise ValueError(f"Unsupported upsample block type: {block_type}")
    if shortcut is None:
        return block
    if shortcut == "duplicating":
        return ResidualBlock(block, ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, factor=2))
    raise ValueError(f"Unsupported upsample shortcut: {shortcut}")


def build_encoder_project_out_block(
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
    shortcut: Optional[str],
) -> nn.Module:
    block = OpSequential(
        [
            build_norm(norm, in_channels),
            build_act(act),
            ConvLayer(in_channels, out_channels, kernel_size=3, use_bias=True, norm=None, act_func=None),
        ]
    )
    if shortcut is None:
        return block
    if shortcut == "averaging":
        return ResidualBlock(block, PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, factor=1))
    raise ValueError(f"Unsupported encoder project_out shortcut: {shortcut}")


def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    block = ConvLayer(in_channels, out_channels, kernel_size=3, use_bias=True, norm=None, act_func=None)
    if shortcut is None:
        return block
    if shortcut == "duplicating":
        return ResidualBlock(block, ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, factor=1))
    raise ValueError(f"Unsupported decoder project_in shortcut: {shortcut}")


def build_decoder_project_out_block(
    in_channels: int,
    out_channels: int,
    factor: int,
    upsample_block_type: str,
    norm: Optional[str],
    act: Optional[str],
) -> nn.Module:
    layers: list[nn.Module] = [build_norm(norm, in_channels), build_act(act)]
    if factor == 1:
        layers.append(ConvLayer(in_channels, out_channels, kernel_size=3, use_bias=True, norm=None, act_func=None))
    elif factor == 2:
        layers.append(build_upsample_block(upsample_block_type, in_channels, out_channels, shortcut=None))
    else:
        raise ValueError(f"Unsupported decoder project_out factor: {factor}")
    return OpSequential(layers)


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        if len(cfg.depth_list) != num_stages:
            raise ValueError("width_list and depth_list must have the same length.")

        project_width = cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1]
        self.project_in = build_encoder_project_in_block(
            cfg.in_channels,
            project_width,
            factor=1 if cfg.depth_list[0] > 0 else 2,
            downsample_block_type=cfg.downsample_block_type,
        )

        stages = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            stage = build_stage_main(
                width=width,
                depth=depth,
                block_type=_stage_value(cfg.block_type, stage_id),
                norm=_stage_value(cfg.norm, stage_id),
                act=_stage_value(cfg.act, stage_id),
                input_width=width,
            )
            if stage_id < num_stages - 1 and depth > 0:
                out_width = cfg.width_list[stage_id + 1] if cfg.downsample_match_channel else width
                stage.append(build_downsample_block(cfg.downsample_block_type, width, out_width, cfg.downsample_shortcut))
            stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(stages)
        self.project_out = build_encoder_project_out_block(
            cfg.width_list[-1],
            2 * cfg.latent_channels if cfg.double_latent else cfg.latent_channels,
            cfg.out_norm,
            cfg.out_act,
            cfg.out_shortcut,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            x = stage(x)
        return self.project_out(x)


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        if len(cfg.depth_list) != num_stages:
            raise ValueError("width_list and depth_list must have the same length.")
        self.project_in = build_decoder_project_in_block(cfg.latent_channels, cfg.width_list[-1], cfg.in_shortcut)

        stages = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage: list[nn.Module] = []
            if stage_id < num_stages - 1 and depth > 0:
                in_width = cfg.width_list[stage_id + 1]
                out_width = width if cfg.upsample_match_channel else in_width
                stage.append(build_upsample_block(cfg.upsample_block_type, in_width, out_width, cfg.upsample_shortcut))
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=_stage_value(cfg.block_type, stage_id),
                    norm=_stage_value(cfg.norm, stage_id),
                    act=_stage_value(cfg.act, stage_id),
                    input_width=width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)],
                )
            )
            stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(stages)
        project_out_width = cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1]
        self.project_out = build_decoder_project_out_block(
            project_out_width,
            cfg.out_channels,
            factor=1 if cfg.depth_list[0] > 0 else 2,
            upsample_block_type=cfg.upsample_block_type,
            norm=cfg.out_norm,
            act=cfg.out_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in reversed(self.stages):
            x = stage(x)
        return self.project_out(x)


__all__ = [
    "ConvPixelUnshuffleDownSampleLayer",
    "PixelUnshuffleChannelAveragingDownSampleLayer",
    "ConvPixelShuffleUpSampleLayer",
    "ChannelDuplicatingPixelUnshuffleUpSampleLayer",
    "EncoderConfig",
    "DecoderConfig",
    "Encoder",
    "Decoder",
]
