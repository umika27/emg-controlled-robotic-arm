import torch
from torch import nn


class GestureCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(8, 24, 5, padding=2), nn.ReLU(),
            nn.Conv1d(24, 48, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(48, classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 8:
            raise ValueError('CNN expects [batch, 8, samples].')
        return self.network(x)
