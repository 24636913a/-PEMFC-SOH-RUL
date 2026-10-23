"""Exact N2 class copies only; no old N2 training/data code. TCN has symmetric padding."""
from typing import Any
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, num_layers: int, dropout: float, **_: Any) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.head(y[:, -1])

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation), nn.GELU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + self.skip(x))

class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, num_layers: int, kernel_size: int, dropout: float, **_: Any) -> None:
        super().__init__()
        blocks = []
        channels = input_dim
        for i in range(num_layers):
            blocks.append(TCNBlock(channels, hidden_dim, kernel_size, 2**i, dropout))
            channels = hidden_dim
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tcn(x.transpose(1, 2)))
