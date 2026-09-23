# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""
Modified from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from models.flow.utils import (
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from models.flow.module import (
    TimestepEmbedSequential,
    Upsample,
    Downsample,
    ResBlock,
    AttentionBlock,
    base2_fourier_features,
)
@dataclass(eq=False)
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """
    
    in_channels: int
    compress_channels: int = -1
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    use_flash_attention: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    with_fourier_features: bool = False
    ignore_time: bool = False
    input_projection: bool = True

    image_size: int = -1  # not used...
    _target_: str = "lib.models.gd_unet.UNetModel"
    
    def _module_init(self):
        super().__init__()
    
    def __post_init__(self):
        self._module_init()
        
        if self.with_fourier_features:
            self.in_channels += 12

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.time_embed_dim = self.model_channels * 4
        if self.ignore_time:
            self.time_embed = lambda x: torch.zeros(
                x.shape[0], self.time_embed_dim, device=x.device, dtype=x.dtype
            )
        else:
            self.time_embed = nn.Sequential(
                linear(self.model_channels, self.time_embed_dim),
                nn.SiLU(),
                linear(self.time_embed_dim, self.time_embed_dim),
            )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(
                self.num_classes + 1, self.time_embed_dim, padding_idx=self.num_classes
            )
        
        self.compress = conv_nd(self.dims, self.in_channels, self.compress_channels, 1) if self.compress_channels!=-1 else torch.nn.Identity()
        self.decompress = zero_module(conv_nd(self.dims, self.compress_channels, self.out_channels, 1)) if self.compress_channels!=-1 else torch.nn.Identity()
        
        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        if self.input_projection:
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(self.dims, self.in_channels if self.compress_channels==-1 else self.compress_channels, ch, 3, padding=1)
                    )
                ]
            )
        else:
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(torch.nn.Identity())]
            )
        
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                            use_flash_attention=self.use_flash_attention
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
                use_flash_attention=self.use_flash_attention
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                            use_flash_attention=self.use_flash_attention
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Upsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1)) if self.compress_channels==-1 
            else conv_nd(self.dims, input_ch, self.compress_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, extra):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.compress(x)
        
        if self.with_fourier_features:
            z_f = base2_fourier_features(x, start=6, stop=8, step=1)
            x = torch.cat([x, z_f], dim=1)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x))

        if self.ignore_time:
            emb = emb * 0.0

        if self.num_classes and "label" not in extra:
            # Hack to deal with ddp find_unused_parameters not working with activation checkpointing...
            # self.num_classes corresponds to the pad index of the embedding table
            extra["label"] = torch.full(
                (x.size(0),), self.num_classes, dtype=torch.long, device=x.device
            )

        if self.num_classes is not None and "label" in extra:
            y = extra["label"]
            assert (
                y.shape == x.shape[:1]
            ), f"Labels have shape {y.shape}, which does not match the batch dimension of the input {x.shape}"
            emb = emb + self.label_emb(y)
        
        h = x
        if "concat_conditioning" in extra:
            h = torch.cat([x, extra["concat_conditioning"]], dim=1)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        h = h.type(x.dtype)
        result = self.out(h)
        
        result = self.decompress(result)
        return result
