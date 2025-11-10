# baseline_sensor/timesnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimesNet(nn.Module):
    def __init__(self, sensor_channels=6, num_classes=14, base_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(sensor_channels, base_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(base_dim),
            nn.ReLU(),
            nn.Conv1d(base_dim, base_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(),
            nn.Conv1d(base_dim * 2, base_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_dim * 4),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_dim * 4, num_classes)

    def forward(self, x):
        x = self.encoder(x)  # [B, C, T]
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
