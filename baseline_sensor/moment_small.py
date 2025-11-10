# baseline_sensor/moment_small.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class MomentSmall(nn.Module):
    def __init__(self, sensor_channels=6, num_classes=14, base_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(sensor_channels, base_dim, 7, padding=3),
            nn.BatchNorm1d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_dim),
            ResidualBlock(base_dim),
            ResidualBlock(base_dim)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
