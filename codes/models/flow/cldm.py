from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from models.flow.module import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepBlock,
    TimestepEmbedSequential,
    Upsample,
)
from models.flow.unet import UNetModel
from models.flow.utils import (
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


class ControlledUNetModel(UNetModel):
    """The pretrained UNet with ControlNet residuals injected into its decoder."""

    def forward(self, x, timesteps, extra, control):
        """Run the frozen UNet while adding ControlNet residuals at each decoder level."""
        with torch.no_grad():
            x = self.compress(x)

            if self.with_fourier_features:
                from models.flow.module import base2_fourier_features

                z_f = base2_fourier_features(x, start=6, stop=8, step=1)
                x = torch.cat([x, z_f], dim=1)

            hs = []
            emb = self.time_embed(
                timestep_embedding(timesteps, self.model_channels).to(x)
            )

            if self.ignore_time:
                emb = emb * 0.0

            if self.num_classes and "label" not in extra:
                extra["label"] = torch.full(
                    (x.size(0),),
                    self.num_classes,
                    dtype=torch.long,
                    device=x.device,
                )

            if self.num_classes is not None and "label" in extra:
                y = extra["label"]
                assert y.shape == x.shape[:1], (
                    f"Labels have shape {y.shape}, which does not match "
                    f"the batch dimension of the input {x.shape}"
                )
                emb = emb + self.label_emb(y)

            h = x
            if "concat_conditioning" in extra:
                h = torch.cat([x, extra["concat_conditioning"]], dim=1)

            for module in self.input_blocks:
                h = module(h, emb)
                hs.append(h)

            h = self.middle_block(h, emb)

        h = h + control.pop()

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        result = self.out(h)
        return self.decompress(result)


@dataclass(eq=False)
class ControlNet(UNetModel):
    """Trainable branch that converts a hint tensor into multi-scale UNet residuals."""

    hint_channel: int = -1

    def __post_init__(self):
        self._module_init()

        self.in_channels = self.hint_channel
        if self.with_fourier_features:
            self.in_channels += 12

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.time_embed_dim = self.model_channels * 4
        if self.ignore_time:
            self.time_embed = lambda x: torch.zeros(
                x.shape[0],
                self.time_embed_dim,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            self.time_embed = nn.Sequential(
                linear(self.model_channels, self.time_embed_dim),
                nn.SiLU(),
                linear(self.time_embed_dim, self.time_embed_dim),
            )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(
                self.num_classes + 1,
                self.time_embed_dim,
                padding_idx=self.num_classes,
            )

        self.compress = (
            conv_nd(
                self.dims,
                self.in_channels,
                self.compress_channels,
                1,
            )
            if self.compress_channels != -1
            else nn.Identity()
        )
        self.decompress = (
            zero_module(
                conv_nd(
                    self.dims,
                    self.compress_channels,
                    self.out_channels,
                    1,
                )
            )
            if self.compress_channels != -1
            else nn.Identity()
        )

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        if self.input_projection:
            input_channels = (
                self.in_channels
                if self.compress_channels == -1
                else self.compress_channels
            )
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(self.dims, input_channels, ch, 3, padding=1)
                    )
                ]
            )
            self.zero_convs = nn.ModuleList([self.make_zero_conv(ch)])
        else:
            input_channels = (
                self.in_channels
                if self.compress_channels == -1
                else self.compress_channels
            )
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(nn.Identity())]
            )
            self.zero_convs = nn.ModuleList(
                [self.make_zero_conv(input_channels)]
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
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
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
                            ch,
                            self.conv_resample,
                            dims=self.dims,
                            out_channels=out_ch,
                        )
                    )
                )
                self.zero_convs.append(self.make_zero_conv(ch))
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
        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        """Create a zero-initialized 1x1 projection for one control feature scale."""
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, timesteps, extra):
        """Encode the condition and return residuals consumed by ``ControlledUNetModel``."""
        x = self.compress(x)

        if self.with_fourier_features:
            from models.flow.module import base2_fourier_features

            z_f = base2_fourier_features(x, start=6, stop=8, step=1)
            x = torch.cat([x, z_f], dim=1)

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels).to(x)
        )

        if self.ignore_time:
            emb = emb * 0.0

        if self.num_classes and "label" not in extra:
            extra["label"] = torch.full(
                (x.size(0),),
                self.num_classes,
                dtype=torch.long,
                device=x.device,
            )

        if self.num_classes is not None and "label" in extra:
            y = extra["label"]
            assert y.shape == x.shape[:1], (
                f"Labels have shape {y.shape}, which does not match "
                f"the batch dimension of the input {x.shape}"
            )
            emb = emb + self.label_emb(y)

        h = x
        if "concat_conditioning" in extra:
            h = torch.cat([x, extra["concat_conditioning"]], dim=1)

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb)
            hs.append(zero_conv(h, emb))

        h = self.middle_block(h, emb)
        hs.append(self.middle_block_out(h, emb))
        return hs


@dataclass(eq=False)
class ControlLDM(UNetModel):
    """Compose a pretrained UNet and trainable ControlNet for conditional generation."""

    hint_channel: int = -1
    control_key: str = "hint"

    def __post_init__(self):
        """Build the paired branches from the shared UNet configuration."""
        self._module_init()
        if self.hint_channel == -1:
            raise ValueError("ControlLDM requires hint_channel to be specified")

        kwargs = {field.name: getattr(self, field.name) for field in fields(UNetModel)}
        self.unet = ControlledUNetModel(**kwargs)

        kwargs["hint_channel"] = self.hint_channel
        self.controlnet = ControlNet(**kwargs)
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def load_pretrained_ldm(self, ldm_ckpt, sd_key="ldm_state_dict"):
        """Load the pretrained UNet branch from the named checkpoint entry."""
        try:
            checkpoint = torch.load(
                ldm_ckpt,
                map_location=torch.device("cpu"),
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            checkpoint = torch.load(
                ldm_ckpt,
                map_location=torch.device("cpu"),
                weights_only=False,
            )

        if sd_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain '{sd_key}': {ldm_ckpt}")
        self.unet.load_state_dict(checkpoint[sd_key], strict=True)
        del checkpoint
        print("LDM checkpoint loaded successfully.")

    @torch.no_grad()
    def load_controlnet_from_unet(self):
        """Initialize matching ControlNet weights from the UNet and zero new channels."""
        unet_sd = self.unet.state_dict()
        controlnet_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()

        for key, control_value in controlnet_sd.items():
            if key in unet_sd:
                unet_value = unet_sd[key]
                if control_value.size() == unet_value.size():
                    init_sd[key] = unet_value.clone()
                else:
                    if control_value.ndim != 4 or unet_value.ndim != 4:
                        raise ValueError(
                            f"Cannot expand non-convolutional ControlNet parameter: {key}"
                        )
                    extra_channels = control_value.size(1) - unet_value.size(1)
                    if extra_channels < 0:
                        raise ValueError(
                            f"ControlNet parameter has fewer input channels than UNet: {key}"
                        )
                    zeros = torch.zeros(
                        (control_value.size(0), extra_channels, *control_value.shape[2:]),
                        dtype=unet_value.dtype,
                        device=unet_value.device,
                    )
                    init_sd[key] = torch.cat((unet_value, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = control_value.clone()
                init_with_scratch.add(key)

        self.controlnet.load_state_dict(init_sd, strict=True)
        print("ControlNet parameters initialized from the pretrained UNet.")
        return init_with_new_zero, init_with_scratch

    def forward(self, x, timesteps, extra):
        """Predict with ControlNet residuals injected into the pretrained UNet."""
        hint = extra[self.control_key]
        control = self.controlnet(hint, timesteps, extra)
        control = [value * scale for value, scale in zip(control, self.control_scales)]
        return self.unet(x, timesteps, extra, control)
