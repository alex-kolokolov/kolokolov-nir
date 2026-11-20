from __future__ import annotations

import torch
from torch import nn


class MaskedRevIN(nn.Module):
    """Window-local RevIN with shared affine parameters and missing-value masks."""

    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # Shared scalars keep RevIN valid when a farm exposes fewer channels.
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def normalize(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mask_f = mask.to(dtype=x.dtype)
        count = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * mask_f).sum(dim=1, keepdim=True) / count
        variance = ((x - mean).square() * mask_f).sum(dim=1, keepdim=True) / count
        std = (variance + self.eps).sqrt()
        normalized = (x - mean) / std
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized * mask_f, (mean, std)

    def denormalize_target(
        self,
        values: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
        target_channel: int = 0,
    ) -> torch.Tensor:
        mean, std = stats
        if self.affine:
            values = (values - self.bias) / self.weight.clamp_min(self.eps)
        target_mean = mean[:, 0, target_channel].unsqueeze(-1)
        target_std = std[:, 0, target_channel].unsqueeze(-1)
        return values * target_std + target_mean

