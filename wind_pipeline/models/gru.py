"""Small GRU with a residual power head; independent of the old Transformer."""
import torch
from torch import nn


class ResidualGRU(nn.Module):
    def __init__(self, channels=9, hidden_size=48, horizons=4):
        super().__init__()
        self.gru = nn.GRU(channels, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, horizons)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor):
        _, hidden = self.gru(x)
        return x[:, -1, :1] + self.head(hidden[-1])
