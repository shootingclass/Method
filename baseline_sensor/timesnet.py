import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    Basic TimesNet Inception-style block for multi-scale temporal modeling
    (Simplified version for sensor embedding)
    """
    def __init__(self, in_channels, out_channels, kernel_set=(3, 5, 7), dilation=1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, out_channels, (1, k), dilation=dilation, padding=(0, k//2))
            for k in kernel_set
        ])
        self.bn = nn.BatchNorm2d(out_channels * len(kernel_set))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 1, C, T]
        out = [conv(x) for conv in self.convs]  # [B, out_channels, C, T] × len(kernel_set)
        out = torch.cat(out, dim=1)
        return self.relu(self.bn(out))


class TimesBlock(nn.Module):
    """
    TimesNet 2D temporal variation modeling block
    (kept residual structure for stability)
    """
    def __init__(self, in_channels, hidden_channels, kernel_set=(3, 5, 7)):
        super().__init__()
        self.inception = InceptionBlock(in_channels, hidden_channels, kernel_set)
        self.conv1x1 = nn.Conv2d(hidden_channels * len(kernel_set), in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.inception(x)
        out = self.conv1x1(out)
        out = self.bn(out)
        return self.relu(out + residual)


class TimesNetEncoder(nn.Module):
    """
    ✅ Sensor-only TimesNet encoder (for Linear Probing)
    Returns: [B, D] feature embedding
    """
    def __init__(self, args, hidden_dim: int = 64, layers: int = 3):
        super().__init__()
        self.embedding_dim = args.embedding_dim
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.blocks = nn.ModuleList([
            TimesBlock(hidden_dim, hidden_dim) for _ in range(layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(hidden_dim, args.embedding_dim)

    def forward(self, x):
        # x: [B, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.global_pool(x)  # [B, hidden_dim, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, hidden_dim]
        emb = self.proj(x)  # [B, D]
        return emb
