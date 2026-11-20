from __future__ import annotations

import math

import torch
from torch import nn

from .revin import MaskedRevIN


class MaskedChannelPatchTST(nn.Module):
    """Shared PatchTST encoder with masked late fusion, Sensor Dropout, and Persistence Gating.

    - Channel-independent encoder handles arbitrary subsets of available channels.
    - Sensor Dropout during training forces resilience against missing optional sensors.
    - Persistence-gated residual head blends neural forecast with Persistence based on local wind volatility.
    """

    def __init__(
        self,
        *,
        lookback: int,
        horizons: int,
        max_channels: int,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        sensor_dropout: float = 0.3,
        target_channel: int = 0,
        wind_channel: int = 1,
    ):
        super().__init__()
        if patch_length > lookback:
            raise ValueError("patch_length cannot exceed lookback")
        self.lookback = lookback
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.target_channel = target_channel
        self.wind_channel = wind_channel
        self.sensor_dropout = sensor_dropout
        self.n_patches = 1 + math.ceil((lookback - patch_length) / patch_stride)
        padded_length = (self.n_patches - 1) * patch_stride + patch_length
        self.right_padding = padded_length - lookback

        self.revin = MaskedRevIN(affine=True)
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        self.channel_embedding = nn.Embedding(max_channels, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.channel_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizons),
        )

        # Persistence gating network:
        # Inputs: wind standard deviation, wind ramp delta across window, mean wind level
        self.gate_net = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, horizons),
            nn.Sigmoid(),
        )
        # Initialize gate bias so initial alpha is moderately conservative (~0.3-0.5)
        nn.init.constant_(self.gate_net[-2].bias, -0.5)
        nn.init.normal_(self.gate_net[-2].weight, std=0.01)

        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        channel_ids: torch.Tensor | None = None,
        p_last: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape != x_mask.shape:
            raise ValueError("x and x_mask must have shape [batch, time, channels]")
        batch, length, channels = x.shape
        if length != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, got {length}")
        if channel_ids is None:
            channel_ids = torch.arange(channels, device=x.device)
        if channel_ids.numel() != channels:
            raise ValueError("channel_ids must contain one id per input channel")

        # Determine reference persistence power P(t)
        if p_last is None:
            p_ref = x[:, -1, self.target_channel].unsqueeze(-1)
        else:
            p_ref = p_last.view(batch, 1)

        # Apply Sensor Dropout during training to optional channels
        active_mask = x_mask
        if self.training and self.sensor_dropout > 0 and channels > 2:
            rand_vals = torch.rand(batch, 1, channels, device=x.device)
            drop_mask = rand_vals < self.sensor_dropout
            # Protect target channel and wind speed channel from dropout
            drop_mask[:, :, self.target_channel] = False
            if self.wind_channel is not None and self.wind_channel < channels:
                drop_mask[:, :, self.wind_channel] = False
            active_mask = active_mask & (~drop_mask)

        normalized, stats = self.revin.normalize(x, active_mask)
        if self.right_padding:
            normalized = nn.functional.pad(normalized, (0, 0, 0, self.right_padding))
            active_mask = nn.functional.pad(active_mask, (0, 0, 0, self.right_padding))

        patches = normalized.unfold(1, self.patch_length, self.patch_stride)
        patch_mask = active_mask.unfold(1, self.patch_length, self.patch_stride).any(dim=-1)
        # unfold: [B, patches, C, patch_length] -> [B, C, patches, patch_length]
        patches = patches.permute(0, 2, 1, 3)
        patch_mask = patch_mask.permute(0, 2, 1)
        tokens = self.patch_projection(patches) + self.position
        tokens = tokens + self.channel_embedding(channel_ids)[None, :, None, :]

        flat = tokens.reshape(batch * channels, self.n_patches, -1)
        flat_valid = patch_mask.reshape(batch * channels, self.n_patches)
        channel_available = flat_valid.any(dim=1)
        # Transformer attention cannot accept rows where every token is masked.
        safe_valid = flat_valid.clone()
        safe_valid[~channel_available, 0] = True
        encoded = self.encoder(flat, src_key_padding_mask=~safe_valid)
        weights = safe_valid.unsqueeze(-1).to(encoded.dtype)
        encoded = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        encoded = encoded.reshape(batch, channels, -1)

        available = channel_available.reshape(batch, channels)
        scores = self.channel_score(encoded).squeeze(-1).masked_fill(~available, -1e4)
        channel_weights = torch.softmax(scores, dim=1)
        fused = (encoded * channel_weights.unsqueeze(-1)).sum(dim=1)
        normalized_forecast = self.head(fused)
        raw_forecast = self.revin.denormalize_target(
            normalized_forecast, stats, target_channel=self.target_channel
        )

        # Persistence Gating computation
        if self.wind_channel is not None and self.wind_channel < channels:
            wind_seq = x[:, :, self.wind_channel]
            wind_std = wind_seq.std(dim=1, keepdim=True)
            wind_ramp = (wind_seq[:, -1:] - wind_seq[:, :1]).abs()
            wind_mean = wind_seq.mean(dim=1, keepdim=True)
            gate_input = torch.cat([wind_std, wind_ramp, wind_mean], dim=-1)
        else:
            gate_input = torch.zeros(batch, 3, device=x.device, dtype=x.dtype)

        alpha = self.gate_net(gate_input)  # [batch, horizons] in [0, 1]
        forecast = (1.0 - alpha) * p_ref + alpha * raw_forecast
        return forecast.clamp_min(0.0)
