import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series"""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))  # [B, C, L]
        x = x.permute(0, 2, 1)            # [B, L, C]
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block"""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearEncoder(nn.Module):
    """
    ✅ Sensor-only DLinear encoder (no classifier)
    Returns: [B, D] embedding vector
    Reference: https://github.com/cure-lab/DLinear
    """
    def __init__(self, args, kernel_size=25):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = 1
        self.num_sensors = args.num_sensors
        self.embedding_dim = args.embedding_dim

        self.decomp = SeriesDecomp(kernel_size)
        self.individual = False

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.num_sensors):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # Projection head (embedding)
        self.proj = nn.Linear(self.pred_len * self.num_sensors, self.embedding_dim)

    def forward(self, x):
        """
        x: [B, L, C]  (Batch, Sequence length, Sensor channels)
        """
        seasonal, trend = self.decomp(x)
        seasonal, trend = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)  # [B, C, L]
        if self.individual:
            seasonal_out, trend_out = [], []
            for i in range(self.num_sensors):
                s = self.Linear_Seasonal[i](seasonal[:, i, :])
                t = self.Linear_Trend[i](trend[:, i, :])
                seasonal_out.append(s.unsqueeze(1))
                trend_out.append(t.unsqueeze(1))
            seasonal_out = torch.cat(seasonal_out, dim=1)
            trend_out = torch.cat(trend_out, dim=1)
        else:
            seasonal_out = self.Linear_Seasonal(seasonal)
            trend_out = self.Linear_Trend(trend)

        out = seasonal_out + trend_out               # [B, C, pred_len]
        out = out.reshape(out.size(0), -1)           # flatten [B, C*pred_len]
        emb = self.proj(out)                         # [B, D]
        return emb
