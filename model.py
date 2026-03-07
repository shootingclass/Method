import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import wandb
import matplotlib.pyplot as plt
from transformers import CLIPVisionModelWithProjection, AutoModel
from peft import LoraConfig, get_peft_model
from einops import repeat
import torch.distributed as dist
from torchvision.transforms.functional import crop, resize
import torchvision
import cv2
import math
from PIL import Image, ImageDraw

from analysis.visualizes import visualize_tsne, visualize_sensor_name, START_INDEX, END_INDEX, visualize_cropped_tensor, denormalize, compute_hungarian_matching, visualize_joint_space, compute_alignment_score, cross_modal_retrieval
from tqdm import tqdm
from method_utils import gather, time_warp


#################################################################



# # Block 클래스는 그대로 둔다고 가정
# class ChannelAttention(nn.Module):
#     def __init__(self, channels, reduction=8):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         # x: [B, C, L]
#         w = x.mean(dim=-1)  # Global avg pooling across time
#         attention = w.clone()
#         w = self.fc(w)      # [B, C]
#         w = w.unsqueeze(-1) # [B, C, 1]
#         return x * (1 + 0.5 * w), attention


# class SensorMotionEncoder(nn.Module):
#     def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
#         super().__init__()
#         # 개별 레이어 정의
#         self.norm1 = torch.nn.GroupNorm(1, sensor_channels)
#         self.block1 = Block(sensor_channels, input_dim, 5)
#         self.block2 = Block(input_dim, input_dim * 2, 3)
#         self.block3 = Block(input_dim * 2, input_dim * 2, 3, 
#                             pool_type="adaptive", embedding_size=32) # Adaptive 출력 길이는 32
#         self.norm2 = torch.nn.GroupNorm(4, input_dim * 2)
#         self.attn = ChannelAttention(sensor_channels)
        
#         # ★★★ GRU input_size 수정 ★★★
#         self.gru = torch.nn.GRU(
#             batch_first=True, 
#             input_size=input_dim * 2, # 이전 레이어 출력 채널과 일치
#             hidden_size=size_embeddings
#         )
        
#         self.ssl_head = torch.nn.Linear(size_embeddings, size_embeddings)
#         self.mmcl_head = torch.nn.Linear(size_embeddings, size_embeddings)
# # --- 추가 변수 ---
#         # ---- Attention 추가 ----
#         self.attn = ChannelAttention(sensor_channels)
#         self.ema_decay = 0.9          # EMA smoothing (0.8 → 0.9 추천)
#         self.attn_scale = 0.5       # attention strength (e.g., 0.5)
#         self.register_buffer("attn_smooth", torch.zeros(sensor_channels))

#         # ---- GRU + head ----
#         self.gru = nn.GRU(batch_first=True, input_size=input_dim * 2, hidden_size=size_embeddings)
#         self.ssl_head = nn.Linear(size_embeddings, size_embeddings)
#         self.mmcl_head = nn.Linear(size_embeddings, size_embeddings)

#     def forward(self, batch, labels=None):
#        # 1️⃣ Attention
#         x, attention = self.attn(batch)  # attention: [B, C]
#         w_mean = attention.mean(dim=0).detach()

#         # EMA smoothing
#         self.attn_smooth = self.ema_decay * self.attn_smooth + (1 - self.ema_decay) * w_mean

#         # Door indices (for Opportunity++)
#         door_idx = [207 - 194, 208 - 194, 209 - 194]

#         # 2️⃣ Regularization factors
#         door_focus = self.attn_smooth[door_idx].mean()
#         door_var = torch.var(self.attn_smooth[door_idx])
#         global_var = torch.var(self.attn_smooth)
#         change_rate = (attention - w_mean.unsqueeze(0)).abs().mean()

#         # 3️⃣ Regularization loss
#         attn_reg = (0.05 * (door_focus.abs()) + 0.01 * global_var + 0.05 * door_var)
#         attn_reg += 0.03 * (1 - door_focus.abs().clamp(max=1))

#         # # 4️⃣ wandb logging
#         # if labels is not None and hasattr(wandb, "log"):
#         #     wandb.log({
#         #         "attention/door1/open_x": self.attn_smooth[door_idx[0]].item(),
#         #         "attention/door1/open_y": self.attn_smooth[door_idx[1]].item(),
#         #         "attention/door1/open_z": self.attn_smooth[door_idx[2]].item(),
#         #         "attention/door_focus_ratio": door_focus.item(),
#         #         "attention/door1/diff_mean": door_var.item(),
#         #         "attention/change_rate": change_rate.item(),
#         #         "attention/global_var": global_var.item(),
#         #         "loss/attn_reg": attn_reg.item(),
#         #     })

#         # --- scaled modulation ---
#         x = x * (1 + self.attn_scale * self.attn_smooth.unsqueeze(0).unsqueeze(-1))
#         x = self.norm1(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x) 
#         x = self.norm2(x) # shape: (B, C = input_dim*2, L = 32)

#         # ★★★ GRU 입력 전 차원 변경 ★★★
#         # (B, C, L) -> (B, L, C)
#         print(x.shape)
#         x = x.permute(0, 2, 1) # shape: (B, 32, input_dim*2)

#         # ★★★ GRU 호출 및 결과 처리 ★★★
#         _, hidden_state = self.gru(x) # (output_seq, hidden_state)
#         emb = hidden_state[0] # 마지막 은닉 상태 (B, hidden_size)

#         ssl_out = self.ssl_head(emb)
#         mmcl_out = self.mmcl_head(emb)
#         out = {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}
#         return out

# # -----------------------------------------------------------
# # 🧩 Temporal Attention Layer
# # -----------------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):  # [B, L, D]
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        return out.mean(dim=1)  # temporal weighted average


# -----------------------------------------------------------
# 🧱 Dilated Temporal Block
# -----------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, pool=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# -----------------------------------------------------------
# 🚀 SensorMotionEncoder (drop-in replacement for SensorEncoder)
# -----------------------------------------------------------
class SensorMotionEncoder(nn.Module):
    def __init__(self, sensor_channels, size_embeddings=128, base_dim=32):
        super().__init__()

        # 1️⃣ Multi-scale dilated convolutions (temporal receptive field 확대)
        self.block1 = TemporalBlock(sensor_channels, base_dim, kernel_size=5, dilation=1)
        self.block2 = TemporalBlock(base_dim, base_dim * 2, kernel_size=3, dilation=2)
        self.block3 = TemporalBlock(base_dim * 2, base_dim * 4, kernel_size=3, dilation=4, pool=False)

        self.norm = nn.GroupNorm(8, base_dim * 4)

        # 2️⃣ GRU + Attention 조합
        self.gru = nn.GRU(
            input_size=base_dim * 4,
            hidden_size=size_embeddings,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attn = TemporalAttention(size_embeddings * 2)

        # 3️⃣ Projection heads
        self.proj = nn.Linear(size_embeddings * 2, size_embeddings)
        self.ssl_head = nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch):
        """
        batch: [B, C, L]  (sensor sequence)
        return: {"emb": motion embedding, "ssl": ssl_out, "mmcl": mmcl_out}
        """
        # Conv feature extraction
        x = self.block1(batch)
        x = self.block2(x)
        x = self.block3(x)
        x = self.norm(x)                # [B, C, L]
        x = x.permute(0, 2, 1)          # [B, L, C]
        self.gru.flatten_parameters()
        # Temporal modeling
        out, _ = self.gru(x)
        attn_out = self.temporal_attn(out)   # [B, D]
        emb = self.proj(attn_out)            # [B, embedding_dim]

        # Heads
        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)

        return {"emb": emb, "ssl": ssl_out, "mmcl": mmcl_out}


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_type="max", embedding_size=32):
        super().__init__()
        if pool_type == "max":
            pool_fn = torch.nn.MaxPool1d(kernel_size=2)
        elif pool_type == "adaptive":
            pool_fn = torch.nn.AdaptiveAvgPool1d(output_size=embedding_size)
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            pool_fn,
        )

    def forward(self, batch):
        return self.net(batch)
    
    
# class SensorModel(nn.Module):
#     def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
#         super().__init__()
#         self.backbone = torch.nn.Sequential(
#             torch.nn.GroupNorm(1, sensor_channels),
#             Block(sensor_channels, input_dim, 5),
#             Block(input_dim, input_dim *2, 3),
#             Block(input_dim *2, input_dim *2, 3, pool_type="adaptive", embedding_size=32),
#             torch.nn.GroupNorm(4, input_dim *2),
#             torch.nn.GRU(
#                 batch_first=True, input_size=input_dim, hidden_size=size_embeddings
#             ),
#         )
#         self.ssl_head = torch.nn.Linear(size_embeddings, size_embeddings)
#         self.mmcl_head = torch.nn.Linear(size_embeddings, size_embeddings)

#     def forward(self, batch):
#         emb = self.backbone(batch)[1][0] # Last hidden state
#         ssl_out = self.ssl_head(emb)
#         mmcl_out = self.mmcl_head(emb)
#         out = {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}
#         return out


# -----------------------------------------------------------
# 🛠️ Transformer용 Positional Encoding (Sinusoidal)
# -----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# -----------------------------------------------------------
# 🚀 Improved SensorTransformerEncoder
# -----------------------------------------------------------
class SensorTransformerEncoder(nn.Module):
    def __init__(self, sensor_channels, size_embeddings=128, base_dim=32, 
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()

        # 1️⃣ CNN Backbone: GRU 모델과 '완전히 동일하게' 맞춤
        # (Block 클래스 대신 TemporalBlock 사용)
        self.block1 = TemporalBlock(sensor_channels, base_dim, kernel_size=5, dilation=1)
        self.block2 = TemporalBlock(base_dim, base_dim * 2, kernel_size=3, dilation=2)
        # 중요: 여기서 pool=False로 설정하여 시퀀스 길이를 보존하거나, 
        # GRU 모델과 똑같이 맞춥니다. (GRU 모델은 block3에서 pool=False였음)
        self.block3 = TemporalBlock(base_dim * 2, base_dim * 4, kernel_size=3, dilation=4, pool=False)

        self.norm = nn.GroupNorm(8, base_dim * 4)
        
        feature_dim = base_dim * 4  # 128

        # 2️⃣ Positional Encoding & Transformer
        # 학습 가능한 PE 대신 Sinusoidal 사용 추천 (데이터 적을 때 유리)
        self.pos_encoder = PositionalEncoding(feature_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-LN이 학습 안정성이 더 높음
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3️⃣ Projection heads
        self.ssl_head = nn.Linear(feature_dim, size_embeddings)
        self.mmcl_head = nn.Linear(feature_dim, size_embeddings)
        self.out_proj = nn.Linear(feature_dim, size_embeddings)

    def forward(self, batch):
        # 1. CNN Feature Extraction
        x = self.block1(batch)
        x = self.block2(x)
        x = self.block3(x)
        x = self.norm(x)              # [B, C, L]
        x = x.permute(0, 2, 1)        # [B, L, C]

        # 2. Apply Positional Encoding
        x = self.pos_encoder(x)       # [B, L, C]

        # 3. Transformer Encoder
        x = self.transformer(x)       # [B, L, C]

        # 4. Pooling Strategy: [CLS] 대신 Mean Pooling (Global Average Pooling)
        # Transformer가 시퀀스의 문맥을 파악한 후, 시간 축으로 평균을 냄
        emb = x.mean(dim=1)           # [B, C]

        # 5. Heads
        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)
        emb = self.out_proj(emb)

        return {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}

class SensorEncoder(nn.Module):
    def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.block1 = Block(sensor_channels, input_dim, 5)
        self.block2 = Block(input_dim, input_dim * 2, 3)
        self.block3 = Block(input_dim * 2, input_dim * 2, 3, pool_type="adaptive", embedding_size=32)
        self.norm = torch.nn.GroupNorm(4, input_dim * 2)

        self.gru = torch.nn.GRU(
            batch_first=True,
            input_size=input_dim * 2,  # GRU 입력은 feature dimension
            hidden_size=size_embeddings
        )
        self.ssl_head = torch.nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = torch.nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch):
        x = self.block1(batch)
        x = self.block2(x)
        x = self.block3(x)
        x = self.norm(x)        # [B, C, L]

        # ✅ GRU가 [B, L, C]를 기대하므로 permute
        x = x.permute(0, 2, 1)  # [B, L, C]

        _, h = self.gru(x)      # h: [1, B, hidden_size]
        emb = h[0]              # [B, hidden_size]

        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)
        return {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}
    
class SensorModel(nn.Module):
    def __init__(self, sensor_appearance_encoder, sensor_motion_encoder, clustering_module):
        super().__init__()
        self.sensor_appearance_encoder = sensor_appearance_encoder
        self.sensor_motion_encoder = sensor_motion_encoder
        self.clustering_module = clustering_module
        self.norm = nn.LayerNorm(256*2)

    def forward(self, batch):
        # z_sensor embedding
        _, features, _ = self.clustering_module(batch, return_features = True)
        sensor_motion_emb = self.encoding_motion(batch)["emb"]
        s_app_norm = F.normalize(features, dim=1)
        s_mot_norm = F.normalize(sensor_motion_emb, dim=1)
        # features는 그래디언트 차단
        with torch.no_grad():
            s_app_norm = F.normalize(features, dim=1)
        z_sensor_online = torch.cat((s_app_norm, s_mot_norm), dim=1)
        # z_sensor_online = torch.cat((features.detach(), sensor_motion_emb), dim=1)

        # z_sensor_online = F.normalize(z_sensor_online, dim=1) * (z_sensor_online.shape[1] ** 0.5 * 0.2)
        # z_sensor_online = self.norm(z_sensor_online)
        return z_sensor_online

    def encoding_appearance(self, batch, labels, return_features, idx):
        # appearance embedding with clustering module
        scores, features, pure_features = self.clustering_module(batch, labels=labels, return_features=return_features, idx=idx)
        return scores, features, pure_features

    
    def encoding_motion(self, batch):
        # motion embedding
        mot_emb = self.sensor_motion_encoder(batch)
        # for emb_type, emb in mot_emb.items():
        #     mot_emb[emb_type] = (emb - emb.mean(dim=0, keepdim=True)) / (emb.std(dim=0, keepdim=True) + 1e-6)
        return mot_emb



#############################################################

# ---------------------------------------------------------------------
# Attention Head (Object Encoder saliency map)
# ---------------------------------------------------------------------
class AttentionHead(nn.Module):
    """Salient region 강조용 간단한 2D attention head."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        return self.sigmoid(self.conv(x))  # [B, 1, H, W]


# ---------------------------------------------------------------------
# Shared CNN Encoder (Scene/Object 공통)
# ---------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """MOSO 스타일의 공유 CNN feature extractor."""
    def __init__(self, in_channels=3, base_dim=64, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 7, stride=2, padding=3, bias=False),  # 224→112
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1, bias=False), # 112→56
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1, bias=False), # 56→28
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim * 4, out_dim, 3, stride=2, padding=1, bias=False), # 28→14
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # [B, out_dim, H', W']

# class MotionEncoder(nn.Module):
#     """
#     Motion Encoder with cosine-consistent projection.
#     강조점:
#       - CosineProj: norm-invariant projection layer
#       - optional cosine regularization for self-consistency
#     """
#     def __init__(self, in_channels=5, base_dim=32, latent_dim=256, use_cosine_proj=True, use_flow=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.latent_dim = latent_dim
#         self.use_flow = use_flow  # Store whether to use optical flow

#         # 3D backbone (temporal gradient feature)
#         self.backbone = nn.Sequential(
#             nn.Conv3d(in_channels, base_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
#             nn.BatchNorm3d(base_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(base_dim, base_dim * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
#             nn.BatchNorm3d(base_dim * 2),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(base_dim * 2, latent_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
#             nn.BatchNorm3d(latent_dim),
#             nn.ReLU(inplace=True),
#         )

#         self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

#         # ✅ projection head 변경
#         if use_cosine_proj:
#             self.proj = CosineProj(latent_dim, latent_dim)
#         else:
#             self.proj = nn.Linear(latent_dim, latent_dim)

#     def forward(self, videos, flows=None):
#         B, T, _, H, W = videos.shape

#         video_diff = videos[:, 1:] - videos[:, :-1]
        
#         if self.use_flow:
#             # WITH flow: MUST provide actual flow data
#             if not isinstance(flows, torch.Tensor):
#                 raise ValueError(
#                     f"MotionEncoder initialized with use_flow=True, but flows is not a tensor! "
#                     f"Got type: {type(flows)}. "
#                     f"Either provide optical flow data or initialize with use_flow=False."
#                 )
            
#             # Adjust flow temporal dimension if needed
#             if flows.shape[1] > video_diff.shape[1]:
#                 flows = flows[:, : video_diff.shape[1]]
#             elif flows.shape[1] < video_diff.shape[1]:
#                 flows = torch.cat([flows, flows[:, -1:, :, :, :]], dim=1)
            
#             x = torch.cat([video_diff, flows], dim=2)  # [B, T-1, 5, H, W]
#         else:
#             # WITHOUT flow: use 3 channels (RGB diff only) - NO ZERO PADDING!
#             x = video_diff  # [B, T-1, 3, H, W]

#         x = x.permute(0, 2, 1, 3, 4)
#         feature_5d = self.backbone(x)
#         feat = self.spatial_pool(feature_5d).squeeze(-1).squeeze(-1)

#         v_motion = feat.mean(dim=2)

#         # projection
#         v_motion = self.proj(v_motion)
#         return v_motion, feature_5d

import torch
import torch.nn as nn
import torchvision

def find_last_linear_in_features(module: nn.Module) -> int:
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("No nn.Linear found in the given module.")
    return last_linear.in_features

class MotionEncoder(nn.Module):
    def __init__(self, latent_dim=256, use_cosine_proj=True, use_flow=False, base_dim=32, in_channels=3, mv_model="mvit_v2_s"):
        super().__init__()
        import torchvision

        if mv_model == "mvit_v2_s":
            weights = torchvision.models.video.MViT_V2_S_Weights.DEFAULT
            backbone = torchvision.models.video.mvit_v2_s(weights=weights)
        else:
            raise ValueError(mv_model)

        # ✅ head가 Sequential이든 아니든 마지막 Linear의 in_features로 dim 추론
        backbone_dim = find_last_linear_in_features(backbone.head)
        backbone.head = nn.Identity()
        self.backbone = backbone

        self.proj = CosineProj(backbone_dim, latent_dim) if use_cosine_proj else nn.Linear(backbone_dim, latent_dim)

    def forward(self, videos, flows=None):
        # videos: [B, T, 3, H, W]
        x = videos.permute(0, 2, 1, 3, 4).contiguous()   # [B, 3, T, H, W]
        feat = self.backbone(x)                          # [B, backbone_dim]
        v_motion = self.proj(feat)
        return v_motion, feat



# --- Cosine Projection Layer ---
class CosineProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        return F.linear(x, w)




# ---------------------------------------------------------------------
# Scene Encoder (Global context)
# ---------------------------------------------------------------------
class SceneEncoder(nn.Module):
    """Global/static appearance representation."""
    def __init__(self, in_dim=256, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # [B, C, H, W] → [B, latent_dim]
        return self.net(x).flatten(1)


# ---------------------------------------------------------------------
# Object Encoder (Local salient appearance)
# ---------------------------------------------------------------------
class ObjectEncoder(nn.Module):
    """Foreground/local salient representation."""
    def __init__(self, in_dim=256, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        self.attn = AttentionHead(latent_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.conv(x)                 # [B, latent_dim, H, W]
        attn = self.attn(feat)              # [B, 1, H, W]
        obj_feat = self.pool(feat * attn).flatten(1)
        return obj_feat
    
    
# ---------------------------------------------------------------------
# VisionModel (Simplified Appearance Encoder + Motion Encoder)
# ---------------------------------------------------------------------
class VisionModel(nn.Module):
    """
    Simplified MOSO: directly extract v_appearance from shared encoder.
    Keeps MotionEncoder for Stage2 alignment.
    """
    def __init__(self, in_channels=3, base_dim=64, latent_dim=256, use_flow=True):
        super().__init__()
        # shared encoder (2D feature extractor)
        self.shared_encoder = SharedEncoder(in_channels, base_dim, out_dim=latent_dim)
        self.scene_encoder = SceneEncoder(in_dim=latent_dim, latent_dim=latent_dim)
        self.object_encoder = ObjectEncoder(in_dim=latent_dim, latent_dim=latent_dim)

        # Scene + Object 결합 projection
        self.proj = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = nn.LayerNorm(latent_dim*2)
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # motion encoder for stage2
        # Use 3 channels (RGB diff) when no flow, 5 channels (RGB diff + flow) with flow
        motion_in_channels = in_channels + 2 if use_flow else in_channels
        self.motion_encoder = MotionEncoder(
            in_channels=motion_in_channels,  # 5 if use_flow else 3
            base_dim=base_dim // 2,
            latent_dim=latent_dim,
            use_flow=use_flow
        )

    def forward(self, video, flows):
        """
        video: [B, T, C, H, W]
        flows: [B, T, 2, H, W]
        returns: dict with v_appearance, v_motion, z_video_online, vis_v
        """

        B, T, C, H, W = video.shape

        # 1️⃣ Shared appearance features
        video_flat = video.view(B * T, C, H, W)
        shared_feat = self.shared_encoder(video_flat)   # [B*T, D, Hf, Wf]
        _, D, Hf, Wf = shared_feat.shape

        # temporal average pooling
        avg_shared_feat_2d = shared_feat.view(B, T, D, Hf, Wf).mean(dim=1)
        shared_feat = shared_feat.view(B, T, D, Hf, Wf).mean(dim=1)

        v_scene = self.scene_encoder(shared_feat)
        v_object = self.object_encoder(shared_feat)

        # ✅ fuse는 입력 1개만 받으므로 cat 먼저!
        fused = torch.cat([v_scene, v_object], dim=1)
        v_appearance = self.fuse(fused)

        v_motion, feature_map_3d = self.motion_encoder(videos = video, flows=flows)
        # v_motion = (v_motion - v_motion.mean(dim=0, keepdim=True)) / (v_motion.std(dim=0, keepdim=True) + 1e-6)

        # detach appearance for z_video_online
        v_app_norm = F.normalize(v_appearance, dim=1)
        v_mot_norm = F.normalize(v_motion, dim=1)
        with torch.no_grad():
            v_app_norm = v_app_norm.detach()
        z_video_online = torch.cat([v_app_norm, v_mot_norm], dim=1) # motion var 균형을 위해 스케일링 반영
        # z_video_online = torch.cat([v_app_norm.detach()*0.5, v_mot_norm*1.5], dim=1)
        # z_video_online = self.norm(z_video_online)
        # z_video_online = F.normalize(z_video_online, dim=1) * (z_video_online.shape[1] ** 0.5 * 0.2)

        # Forward
        # z_video_online = self.fusion(v_appearance.detach(), v_motion)
        return {
            "v_appearance": v_appearance,
            "v_motion": v_motion,
            "z_video_online": z_video_online,
            "motion_feature_map": feature_map_3d,
            "spatial_feature_map": avg_shared_feat_2d
        }


# --- 3. 메모리 뱅크 관리자 ---
class ClusteringManager(nn.Module):
    def __init__(self, num_clusters, feature_dim, initial_global_threshold, momentum=0.9, temperature=0.1, device='cuda', local_rank=0, min_cluster_size=30):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.local_rank = local_rank

        if self.local_rank == "0":
             # Rank 0에서만 임시 크기로 생성 (KMeans 전까지)
             self.feature_bank = torch.zeros((10000, feature_dim), dtype=torch.float32) # 모든 gpu에서 매 스텝마다 feature_bank는 동기화 (update_samples_memory 참조)
        else:
             # 다른 Rank는 None으로 두는 것이 메모리 절약에 유리하나,
             # DDP에서 속성이 없으면 문제가 되므로, 명시적으로 None으로 설정합니다.
             self.feature_bank = None
        
        self.label_bank = None # 모든 Rank가 None으로 시작, 모든 gpu에서 매 스텝마다 label_bank는 동기화 (update_samples_memory 참조)

        # 각 클러스터별 동적 거리 임계값을 저장할 버퍼
        # 초기값은 hparams의 전역 임계값(fallback)으로 설정
        thresholds_tensor = torch.full(
            (num_clusters,), 
            float(initial_global_threshold), 
            device=device,
            dtype=torch.float32
        )
        self.register_buffer('distance_thresholds', thresholds_tensor)

        self.initialized = False
        # 클러스터 중심점(centroids) 초기화 - 더 넓게 분포되도록 초기화
        # 각 차원마다 균등 분포를 사용하여 더 잘 분산되도록 함
        centroids = torch.rand(num_clusters, feature_dim, device=device) * 2.0 - 1.0  # [-1, 1] 범위의 균등 분포. 모든 gpu에서 cetroids는 broadcast 받음 (update_centroids 참조)
        
        # 정규화를 통해 모든 중심점이 단위 구에 있도록 함
        # self.centroids = F.normalize(self.centroids, dim=1)
        
        # 직교성을 높이기 위한 추가 처리
        # QR 분해를 통해 직교 벡터 얻기
        if num_clusters <= feature_dim:  # 클러스터 수가 차원보다 작거나 같을 때만 가능
            q, r = torch.linalg.qr(centroids.t())  # 직교 행렬 Q 얻기
            centroids = q[:, :num_clusters].t()  # 직교 벡터로 중심점 설정
        self.register_buffer('centroids', centroids)
        
        # 메모리 뱅크 초기화 여부를 추적
        self.memory_initialized = False
        
        # 클러스터 재분배 후 메모리 뱅크 업데이트 지연을 위한 플래그
        self.pending_memory_update = False
        
        # 빈 클러스터 감지와 재할당을 위한 임계값
        self.min_cluster_size = min_cluster_size  # 이 값보다 작으면 비어있다고 간주
        
        # 가중치 계산을 위한 상수
        self.class_weight_power = 0.5  # 클러스터 크기에 적용할 거듭제곱

        self.mapping = None

        self.kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
        # 전체 데이터셋의 pseudo label을 계산하고 메모리 뱅크에 저장하는 함수
        self.cluster_stats = {}  # ✅ 초기화 추가

    @torch.no_grad()
    def _compute_centroids_idx(self, cinds):
        """Compute a few centroids."""
        assert self.local_rank == "0"
        num = len(cinds)
        centroids = torch.zeros((num, self.feature_dim), dtype=torch.float32)
        for i, c in enumerate(cinds):
            idx = np.where(self.label_bank.cpu().numpy() == c)[0]
            centroids[i, :] = self.feature_bank[idx, :].mean(dim=0)
        return centroids

    def _compute_centroids(self):
        """Compute all non-empty centroids."""
        assert self.local_rank == "0"
        label_bank_np = self.label_bank.cpu().numpy()
        argl = np.argsort(label_bank_np)
        sortl = label_bank_np[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(label_bank_np))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        centroids = self.centroids.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            centroids[i, :] = self.feature_bank[argl[st:ed], :].mean(dim=0)
        return centroids

    def update_samples_memory(self, idx: torch.Tensor,
                              feature: torch.Tensor):
        """Update samples memory."""
        assert self.initialized
        # print(f"[{self.local_rank}] Updating samples memory for {idx.shape[0]} samples.")
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize
        valid_mask = idx != -1
        idx = idx[valid_mask]
        feature = feature[valid_mask]

        if idx.numel() == 0:
            return torch.tensor(0.0, device=feature.device)
        idx = gather(idx)
        feature_norm = gather(feature_norm)
        
        idx = idx.cpu()
        if self.local_rank == "0":
            feature_old = self.feature_bank[idx, ...].cuda()
            feature_new = (1 - self.momentum) * feature_old + \
                self.momentum * feature_norm
            feature_norm = feature_new / (
                feature_new.norm(dim=1).view(-1, 1) + 1e-10)
            self.feature_bank=self.feature_bank.cuda()
            self.feature_bank[idx, ...] = feature_norm
        dist.barrier()
        dist.broadcast(feature_norm, src=0) 
        # compute new labels

        # similarity_to_centroids = self.compute_similarity_scores(feature_norm.permute(1, 0))
        feature_norm = feature_norm.permute(1, 0)
        centroids_norm = F.normalize(self.centroids, dim=1)
        similarity_to_centroids = torch.mm(centroids_norm,
                                           feature_norm)  # CxN
        newlabel = similarity_to_centroids.argmax(dim=0)  # cuda tensor
        # self.label_bank = self.label_bank.cuda()
        change_ratio = (newlabel != self.label_bank[idx]
                        ).sum().float().cuda() / float(newlabel.shape[0])
        self.label_bank[idx] = newlabel.cuda().clone()  # all gpu have the same label_bank
        # print("update_samples_memory", change_ratio)
        return change_ratio

    @torch.no_grad()
    def update_centroids_memory(self):
        """Update centroids memory."""
        if self.local_rank == "0":
            center = self._compute_centroids()
            self.centroids.copy_(center)
        dist.broadcast(self.centroids, src=0)
        print(f"[{self.local_rank}] Broadcasted centroids shape: {self.centroids}")

    @torch.no_grad()
    def deal_with_small_clusters(self):
        """
        Gather all label_banks, perform clustering logic on rank 0,
        and broadcast the updated label_bank and centroids back to all ranks if needed.
        """        

        # 1. Rank 0 에서만 모든 계산을 수행합니다.
        if self.local_rank == "0":
            # self.label_bank와 self.feature_bank는 모두 동기화 되어있음. rank 0에서 small_clusters를 안전하게 계산
            global_histogram = np.bincount(
                self.label_bank.cpu().numpy(), minlength=self.num_clusters)
            small_clusters = np.where(global_histogram < self.min_cluster_size)[0].tolist()

            if len(small_clusters) == 0:
                # 변경 사항이 없으면 pass.
                pass
            else:
                print(f'[Rank 0] Dealing with {len(small_clusters)} small clusters.')

                # 재할당 로직 수행 (모든 데이터가 Rank 0에 있으므로 동기화 불필요)
                for s in small_clusters:
                    label_bank_np = self.label_bank.cpu().numpy()
                    idx = np.where(label_bank_np == s)[0]
                    if len(idx) == 0:
                        continue
                    
                    # feature_bank도 모든 GPU에 걸쳐 동일한 복사본이 있어야 합니다.
                    # (만약 아니라면 feature_bank도 gather가 필요합니다)
                    inclusion = np.setdiff1d(np.arange(self.num_clusters), np.array(small_clusters), assume_unique=True)
                    inclusion_tensor = torch.from_numpy(inclusion).cuda()

                    # feature_bank에서 idx에 해당하는 부분만 가져와야 합니다.
                    # feature_bank가 분산되어 있다면, 이 부분도 수정이 필요합니다.

                    target_idx = torch.mm(
                        self.centroids[inclusion_tensor, :],
                        self.feature_bank[idx, :].cuda().permute(1, 0)
                    ).argmax(dim=0)
                    
                    target = inclusion_tensor[target_idx]
                    self.label_bank[idx] = target.cuda()
                # --- 2단계: ⭐️ 이제 비워진 클러스터를 재활용합니다. ⭐️---
                # _redirect_empty_clusters가 내부적으로 centroid 업데이트와 broadcast를 처리합니다.
                self._redirect_empty_clusters(small_clusters)

        print("label_bank device:", self.label_bank.device)
        dist.broadcast(self.centroids, src=0)
        dist.broadcast(self.label_bank, src=0)
        dist.broadcast(self.feature_bank, src=0)
        # 3. Rank 0은 계산된 centroids를 반환, 나머지는 현재 자신의 centroids를 반환
        return 

    @torch.no_grad()
    def _partition_max_cluster(self, max_cluster: np.ndarray):
        """Deterministic split: closest 50% stay, farthest 50% go to new cluster."""
        assert self.local_rank == "0"
        max_cluster_idx = np.where(self.label_bank.cpu().numpy() == max_cluster)[0]
        assert len(max_cluster_idx) >= 2

        # (1) Extract features
        max_cluster_features = self.feature_bank[max_cluster_idx, :]  # [N_c, D]
        if np.any(np.isnan(max_cluster_features.cpu().numpy())):
            raise Exception('Has nan in features.')

        # (2) Compute cluster centroid
        centroid = max_cluster_features.mean(dim=0, keepdim=True)  # [1, D]

        # (3) Compute distance to centroid (L2 norm)
        distances = torch.norm(max_cluster_features - centroid, dim=1)  # [N_c]

        # (4) Sort by distance
        sorted_indices = torch.argsort(distances)  # ascending (near → far)

        # (5) Split deterministically by median (50%)
        mid = len(sorted_indices) // 2
        sub_cluster1_idx = max_cluster_idx[sorted_indices[:mid].cpu()]   # near → stay (old cluster)
        sub_cluster2_idx = max_cluster_idx[sorted_indices[mid:].cpu()]   # far  → new cluster

        # (6) (Optional) check empty safeguard
        if len(sub_cluster1_idx) == 0 or len(sub_cluster2_idx) == 0:
            print("Warning: deterministic partition failed (empty subset). Forcing equal split.")
            sub_cluster1_idx = max_cluster_idx[:len(max_cluster_idx)//2]
            sub_cluster2_idx = max_cluster_idx[len(max_cluster_idx)//2:]

        return sub_cluster1_idx, sub_cluster2_idx


    @torch.no_grad()
    def _redirect_empty_clusters(self, empty_clusters: np.ndarray):
        """Re-direct empty clusters."""
        assert self.local_rank == "0"
        print("empty_clusters", empty_clusters)
        for e in empty_clusters:
            assert (self.label_bank.cpu().numpy() != e).all().item(), \
                f'Cluster #{e} is not an empty cluster.'
            
             # 1. 가장 큰 클러스터를 찾습니다.
            max_cluster = np.bincount(self.label_bank.cpu().numpy(), minlength=self.num_clusters).argmax().item()
            
            # 2. 가장 큰 클러스터를 둘로 분할합니다.
            sub_cluster1_idx, sub_cluster2_idx = self._partition_max_cluster(max_cluster)

            if sub_cluster1_idx is None:
                continue

            # 3. 분할된 그룹 중 하나를 비어있던 클러스터 'e'에 할당합니다.
            self.label_bank[torch.from_numpy(sub_cluster2_idx)] = e
            print(f"max cluster {max_cluster} devided into 2 area and one of them is assigned into {e})")
            
            # 4. 변경된 두 클러스터(max_cluster, e)의 중심점을 다시 계산하고 모든 GPU에 전파합니다.
            #    (이제 이 함수는 centroid 동기화만 책임집니다. label_bank 동기화는 호출한 쪽에서 처리합니다.)
            cinds=[max_cluster, e]
            center = self._compute_centroids_idx(cinds)
            self.centroids[
                torch.LongTensor(cinds).cuda(), :] = center.cuda()
        print("Redirected empty clusters and updated centroids.")


    def compute_similarity_scores(self, features):
        """특징과 중심점 간의 유사도 점수를 계산합니다."""
        # 코사인 유사도 계산 (L2 정규화 후 내적)
        features_norm = F.normalize(features, dim=1)
        centroids_norm = F.normalize(self.centroids, dim=1)
        similarity = torch.mm(features_norm, centroids_norm.t())
        
        # 온도 파라미터 적용
        return similarity / self.temperature
        
    @torch.no_grad()
    def compute_class_weights(self):
        """클러스터 크기에 근거한 클래스 가중치 및 통계치 계산."""
        # ----------------------------------------
        # 1. 클러스터별 샘플 수 (size histogram)
        # ----------------------------------------
        histogram = np.bincount(
            self.label_bank.cpu().numpy(), minlength=self.num_clusters)
        cluster_size = torch.tensor(histogram, device=self.centroids.device)
        # print("Cluster sizes:", cluster_size)
        normalized_sizes = cluster_size + 1e-8

        # ----------------------------------------
        # 2. 거리 기반 통계치 (mean, std, quantiles)
        # ----------------------------------------
        if hasattr(self, "feature_bank") and hasattr(self, "centroids"):
            # feature_bank: [N, D]
            # label_bank: [N]
            features = self.feature_bank.to(self.centroids.device)
            labels = self.label_bank.to(self.centroids.device)

            cluster_stats = {}
            for k in range(self.num_clusters):
                mask = labels == k
                if mask.any():
                    feats_k = features[mask]
                    centroid_k = self.centroids[k].unsqueeze(0)
                    # distances = torch.norm(feats_k - centroid_k, p=2, dim=1)
                    distances = torch.sqrt(((feats_k - centroid_k) ** 2).sum(dim=1))

                    mean_d = distances.mean().item()
                    std_d = distances.std().item()
                    q25 = torch.quantile(distances, 0.25).item()
                    q50 = torch.quantile(distances, 0.50).item()
                    q75 = torch.quantile(distances, 0.75).item()
                    all = torch.quantile(distances, 1.00).item()
                    cluster_stats[k] = {
                        "size": int(mask.sum().item()),
                        "mean": mean_d,
                        "std": std_d,
                        "q25": q25,
                        "q50": q50,
                        "q75": q75,
                        "all": all*2, # all samples are good.
                    }
                else:
                    cluster_stats[k] = {
                        "size": 0,
                        "mean": -1,
                        "std": -1,
                        "q25": -1,
                        "q50": -1,
                        "q75": -1,
                        "all": -1,
                    }

            # ✅ 기록용으로 저장
            self.cluster_stats = cluster_stats
            # if torch.distributed.get_rank() == 0:
            #     print("📊 Cluster distance stats:")
            #     for k, s in cluster_stats.items():
            #         print(f"  {k:02d}: size={s['size']:>5}, mean={s['mean']:.4f}, std={s['std']:.4f}, q75={s['q75']:.4f}")
        else:
            print("⚠️ Warning: feature_bank or centroids not found — cluster stats skipped.")

        # ----------------------------------------
        # 3. 클래스 가중치 계산 (기존 로직 유지)
        # ----------------------------------------
        weights = 1.0 / torch.pow(normalized_sizes, self.class_weight_power)

        # 너무 큰 편차 방지
        max_weight = weights.max()
        min_weight = weights.min()
        if max_weight > min_weight * 10:
            weights = torch.clamp(weights, min=max_weight / 10)

        weights = weights / weights.sum() * self.num_clusters
        return weights


# --- 4. Clustering 모델 ---
class ClusteringModule(nn.Module):
    def __init__(self, encoder, embedding_dim, num_sensors, num_clusters, datamodule, top_k=1, prototype_cache_dir="./cache", dataset_name="custom_dataset", min_cluster_size=30, mid_label=True):
        super().__init__()
        # 딥러닝 백본 선택
        self.encoder = encoder
        self.top_k = top_k
        self.local_rank = os.environ.get("LOCAL_RANK", "0")
        # Clustering 관리자
        self.clustering_manager = ClusteringManager(num_clusters=num_clusters, initial_global_threshold=1.5, feature_dim=embedding_dim, local_rank=self.local_rank, min_cluster_size=min_cluster_size)
        self.projection_layer = nn.Linear(num_sensors, embedding_dim)
        self.epoch = 0
        self.datamodule = datamodule
        self.centroids_update_interval = 1
        self.deal_with_small_clusters_interval = 1
        self.cls_head = nn.Linear(embedding_dim, num_clusters)
        self.attention_weight = nn.Parameter(torch.randn(num_sensors))
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 0과 1 사이의 가중치(알파)를 출력
        )
        self.prototype_cache_dir = prototype_cache_dir
        self.dataset_name = dataset_name
        self.mid_label = mid_label

   # 🌟🌟🌟 Broadcast 로직을 분리한 헬퍼 함수 🌟🌟🌟
    def broadcast_prototypes(self, rank, world_size, device):
        print(f"[{rank}] Broadcasting results from rank 0...", device)
        
        broadcast_device = torch.device(device) 
        num_total_samples = len(self.train_dataloader.dataset)
        feature_dim = self.clustering_manager.feature_dim
        
        # Broadcast를 위해 GPU 텐서 준비 및 할당
        if rank == "0":
            # Rank 0: CPU 텐서를 GPU로 옮겨서 Broadcast
            centroids_gpu = self.clustering_manager.centroids # 이미 register_buffer로 GPU에 있을 수 있음
            label_bank_gpu = self.clustering_manager.label_bank.to(broadcast_device)
            feature_bank_gpu = self.clustering_manager.feature_bank.to(broadcast_device)
        else:
            # Rank != 0: 받을 GPU 텐서를 미리 할당
            centroids_gpu = self.clustering_manager.centroids # (이미 register_buffer이므로 존재)
            label_bank_gpu = torch.empty(num_total_samples, dtype=torch.long, device=broadcast_device)
            feature_bank_gpu = torch.empty(num_total_samples, feature_dim, dtype=torch.float32, device=broadcast_device)
        
        # 1. centroids 전파
        dist.broadcast(centroids_gpu, src=0)
        # 2. label_bank 전파 (GPU 텐서 사용)
        dist.broadcast(label_bank_gpu, src=0)
        # 3. feature_bank 전파 (GPU 텐서 사용)
        dist.broadcast(feature_bank_gpu, src=0)
        # print(f"[{rank}] Broadcasted label_bank shape: {label_bank_gpu.shape}")
        # print(f"[{rank}] Broadcasted feature_bank shape: {feature_bank_gpu.shape}")
        
        # 4. Rank != 0은 Broadcast된 GPU 텐서를 자신의 CPU 뱅크로 복사 및 상태 업데이트
        if rank != "0":
            # 🚨 주의: Broadcast된 결과를 자신의 feature_bank 멤버 변수에 할당합니다.
            self.clustering_manager.label_bank = label_bank_gpu
            self.clustering_manager.feature_bank = feature_bank_gpu
            self.clustering_manager.initialized = True
            
        print(f"[{rank}] Broadcasting complete.")


    @torch.no_grad()
    def init_prototypes_with_data(self, device, num_clusters):
        self.eval()
        # lazy evaluation
        self.train_dataloader = self.datamodule.val_dataloader()
        
        rank = self.local_rank
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank_str = str(rank) # Rank 비교를 문자열로 통일

        # 🌟🌟🌟 1. 초기화 결과 캐시 로드 시도 (Rank 0에서만) 🌟🌟🌟
        dataset_size = len(self.train_dataloader.dataset)
        cache_filename = f"prototypes_{self.dataset_name}_N{dataset_size}_C{num_clusters}.pt" 
        prototype_cache_path = os.path.join(self.prototype_cache_dir, cache_filename)
        
        # ⚠️ (중요) Cache 성공 시, 나머지 Rank는 Broadcast를 기다리고 있어야 합니다.
        cache_hit = False
        if rank_str == "0" and os.path.exists(prototype_cache_path):
            try:
                # 🚨 UnpicklingError 방지: PIL.Image.Image 등이 저장되지 않았다고 가정하거나,
                # torch.load(..., weights_only=False)를 사용해야 할 수 있습니다.
                print(f"[{rank}] Attempting to load prototypes from cache: {prototype_cache_path}")
                
                cached_data = torch.load(prototype_cache_path)
                
                self.clustering_manager.centroids.copy_(cached_data['centroids'].to(device))
                self.clustering_manager.label_bank = cached_data['label_bank'].to(device)
                self.clustering_manager.feature_bank = cached_data['feature_bank'].to(device)
                self.clustering_manager.initialized = True
                cache_hit = True
                print(f"[{rank}] Prototypes loaded successfully from cache.")
                
            except Exception as e:
                print(f"[{rank}] Cache load failed ({e}). Proceeding with feature extraction.")
                if os.path.exists(prototype_cache_path): os.remove(prototype_cache_path)
                pass # 실패 시, 아래의 K-Means 로직으로 진행

        # 🌟🌟🌟 2. Rank 0의 Cache Hit 상태를 다른 Rank에 알립니다. 🌟🌟🌟
        # Rank 0에서 bool을 tensor로 변환하여 Broadcast
        cache_hit_tensor = torch.tensor([cache_hit], dtype=torch.bool, device=device)
        if world_size > 1:
             dist.broadcast(cache_hit_tensor, src=0)
        cache_hit = cache_hit_tensor.item()
        
        # Rank 0이 캐시에 성공했으면, K-Means 루프를 건너뛰고 바로 Broadcast로 이동합니다.
        if not cache_hit:
            
            print(f"[{rank}] Collecting features for KMeans...")
                
            # --- 2-2. 모든 Rank가 특징 추출에 참여 ---
            local_features = []
            local_idx = []
            
            for i, (videos, sensors, labels, sample_ids, _) in enumerate(self.train_dataloader):
                # print("initializing prototypes - processing batch", i, "on rank", rank)
                sensors = sensors.to(device)
                idx, _ = sample_ids
                idx = idx.clone().to(dtype=torch.long, device=device)
                
                _, features, _ = self(sensors, return_features=True)
                # ... (get_representative_sensor_feature 및 projection_layer 로직) ...
                
                local_features.append(features) # GPU 상태로 유지
                local_idx.append(idx) # GPU 상태로 유지
                print(f"[{rank}] Processed batch {i+1}/{len(self.train_dataloader)}")
            
            local_features_tensor = torch.cat(local_features, dim=0)
            local_idx_tensor = torch.cat(local_idx, dim=0)

            # --- 2-3. All-Gather로 전체 특징과 ID 복제 ---
            all_features_gpu = gather(local_features_tensor)
            all_idx_gpu = gather(local_idx_tensor)
            
            # --- 3. Rank 0에서만 K-Means 실행 및 초기화 ---
            if rank_str == "0":
                print(f"[{rank}] Running KMeans...")
                
                # all_features_cpu = all_features_gpu.cpu()
                all_idx_cpu = all_idx_gpu.cpu()
                
                num_total_samples = len(self.train_dataloader.dataset)
                feature_bank_temp = torch.empty(num_total_samples, self.clustering_manager.feature_dim, device="cuda")
                feature_bank_temp[all_idx_cpu, ...] = all_features_gpu # ID를 사용한 최종 재배치
                
                self.clustering_manager.feature_bank = feature_bank_temp # CPU Feature Bank 할당
                    
                all_features_cpu_numpy = self.clustering_manager.feature_bank.cpu().numpy()
                kmeans = self.clustering_manager.kmeans.fit(all_features_cpu_numpy)
                
                prototypes_gpu = torch.from_numpy(kmeans.cluster_centers_).to(device)
                self.clustering_manager.centroids.copy_(F.normalize(prototypes_gpu, dim=-1))
                initial_labels = torch.from_numpy(kmeans.labels_).long()
                self.clustering_manager.label_bank = initial_labels.cuda()
                self.clustering_manager.initialized = True
                
                # 🌟🌟🌟 4. K-Means 완료 후, Rank 0에서 캐시 저장 🌟🌟🌟
                print(f"[{rank}] Saving new prototypes to cache...")
                temp_path = prototype_cache_path + ".tmp"
                
                data_to_save = {
                    'centroids': self.clustering_manager.centroids.cpu().clone(),
                    'label_bank': self.clustering_manager.label_bank.clone(),
                    'feature_bank': self.clustering_manager.feature_bank.clone(),
                }
                
                os.makedirs(os.path.dirname(prototype_cache_path), exist_ok=True) 
                try:
                    torch.save(data_to_save, temp_path)
                    os.rename(temp_path, prototype_cache_path) # Atomic write
                    print(f"[{rank}] Prototypes saved to cache successfully.")
                except Exception as e:
                    print(f"[{rank}] ERROR during atomic cache save: {e}")
                    if os.path.exists(temp_path): os.remove(temp_path)

        # --- 5. 모든 Rank에 결과 전파 (캐시 여부와 관계없이 실행) ---
        if world_size > 1:
            self.broadcast_prototypes(rank_str, world_size, device)

        self.train()
        return self.clustering_manager.centroids.clone()

    def forward(self, x, idx=None, labels=None, return_features=False, step="train"):
        features = self.encoder(x)["emb"]
        if step == "train" or step == "val":

            # 클러스터 유사도 점수 계산
            representative_feature = self.get_representative_sensor_feature(x, labels, num_total_sensors=END_INDEX-START_INDEX+1, top_k=self.top_k, id=idx)
            if labels is not None:
                for i in range(len(labels)):
                    # if labels[i] in [2,3,7]:
                    ranges = representative_feature
                        # ranges = torch.quantile(torch.abs(x[i]), q=0.99, dim=1)
                    # print(f"id: {idx[i]}, labels: {labels[i]}, max_pooling {ranges[i]}")
            representative_feature = self.projection_layer(representative_feature)
            # alpha = self.gate(features)
            alpha=1
            # if self.local_rank == "0" and labels is not None:
                # wandb.l   og("alpha", alpha, labels)
            features = features + alpha * representative_feature
            # features = representative_feature
            
        similarity_scores = self.clustering_manager.compute_similarity_scores(features)
        # distance from centroids
        dist_similarity_scores = torch.cdist(self.clustering_manager.centroids, features)
        # print("dist_similarity_scores:", dist_similarity_scores)
        # similarity_scores = self.cls_head(features)
        # if labels is not None:
        #     for i in range(len(labels)):
        #         if labels[i] in [2,3,7]:
        #             ranges = torch.quantile(torch.abs(x[i]), q=0.99, dim=1)
        #             print(f"id: {idx[i]}, labels: {labels[i]}, max_pooling {ranges}")

        if return_features:
            return similarity_scores, features, features - alpha * representative_feature
        return similarity_scores
    
    @torch.no_grad()
    def get_pseudo_labels(self, idx):
        """(개선된 버전) 메모리 뱅크에서 pseudo label을 효율적으로 조회합니다."""
        # sample_ids는 Dataset에서 온 정수 인덱스의 '리스트'라고 가정
        
        # 1. label_bank가 있는 디바이스 정보를 가져옵니다.
        device = self.clustering_manager.label_bank.device
        
        # 2. 파이썬 리스트를 모델과 같은 디바이스의 텐서로 변환합니다.
        print(idx)
        sids_tensor = idx.detach().clone().to(dtype=torch.long, device=device)
        
        # 3. 텐서 인덱싱을 이용해 한 번에 모든 레이블을 가져옵니다. (훨씬 빠름)
        #    label_bank가 (N,) 크기라면, sids_tensor의 각 값을 인덱스로 사용하여
        #    해당 위치의 레이블들을 한 번에 조회합니다.
        pseudo_labels = self.clustering_manager.label_bank[sids_tensor]
        
        return pseudo_labels
    
        # --- 1. 규칙 기반 특징 추출기 ---
    def get_representative_sensor_feature(self, imu_batch, labels, num_total_sensors=97, top_k=1, id=None):
        """
        각 샘플에서 신호 변화가 가장 큰 센서를 찾아 원-핫 벡터로 만듭니다.
        imu_batch: (B, C, L) 형태의 텐서
        """
        ranges = torch.quantile(torch.abs(imu_batch), q=0.99, dim=2)
        return ranges
        # 1. 각 채널/시퀀스별로 절댓값 계산
        abs_imu = torch.abs(imu_batch)

        # 2. 시간 축(dim=2)을 따라 각 채널의 99% 백분위수 *값* 계산
        #    keepdim=True로 설정하면 이후 비교/계산을 위해 차원 유지 (B, C, 1)
        q99_values = torch.quantile(abs_imu, q=0.99, dim=2, keepdim=True)

        # 3. 각 시간 스텝의 절댓값이 q99_values와 얼마나 차이나는지 계산
        #    quantile 값 자체가 데이터에 없을 수 있으므로, 가장 가까운 값을 찾음
        diff_to_q99 = torch.abs(abs_imu - q99_values)

        # 4. 시간 축(dim=2)에서 차이가 가장 작은 값의 *인덱스* 찾기
        indices_closest_to_q99 = torch.argmin(diff_to_q99, dim=2) # shape: (B, C)

        # 5. 찾은 인덱스를 사용하여 원본 imu_batch에서 값 가져오기 (부호 포함)
        #    torch.gather를 사용하기 위해 인덱스 텐서 차원 추가 (B, C) -> (B, C, 1)
        indices_for_gather = indices_closest_to_q99.unsqueeze(-1)

        #    dim=2 (시간 축)을 따라 해당 인덱스의 값을 선택
        selected_values_with_sign = torch.gather(imu_batch, dim=2, index=indices_for_gather) # shape: (B, C, 1)

        # 6. 마지막 차원 제거 (필요하다면)
        selected_values_with_sign = selected_values_with_sign.squeeze(-1) # shape: (B, C)
        return selected_values_with_sign
        # 각 채널(센서)의 분산 계산 (max - min)
        ranges = torch.quantile(torch.abs(imu_batch), q=0.99, dim=2)
        # print("max pooling: ", ranges)
        # ranges = torch.max(torch.abs(imu_batch), dim=2).values
        # ranges = torch.mean(torch.abs(imu_batch), dim=2)
        # return ranges
        # ranges = torch.var(imu_batch, dim=2)
        # 가장 분산이 큰 센서의 인덱스 찾기 (분산이 0인것 제외)
        # return ranges
        weighted_features = ranges
        # min_range, _ = torch.min(ranges, dim=1, keepdim=True)
        # max_range, _ = torch.max(ranges, dim=1, keepdim=True)
        # weighted_features = (ranges - min_range) / (max_range - min_range + 1e-8)
        # 정규화 x
        # 3. Top-K에 해당하지 않는 값들을 0으로 마스킹
        # 가장 큰 Top-K 값만 남기고 나머지는 0으로 만들기 위한 마스크 생성
        _, top_indices = torch.topk(weighted_features, k=top_k, dim=1)
        mask = torch.zeros_like(weighted_features)
        mask.scatter_(1, top_indices, 1)
            # 센서 이름 출력
        # if labels is not None and id is not None:
        #     visualize_sensor_name(top_indices, labels, id)

        # 4. 마스크를 적용하여 최종 특징 생성
        final_rule_feature = weighted_features * mask
        # print(f"labels {labels}, final {final_rule_feature}")

        return final_rule_feature
    
    def augment_imu_data(self, imu_data):
        return time_warp(imu_data)
    
    def update_epoch(self, epoch):
        self.epoch = epoch
   
    @torch.no_grad()
    def evaluate(self, outputs):
        self.eval()

        # 1) gather
        features = gather(torch.cat([x['features'] for x in outputs])).cpu().numpy()
        labels = gather(torch.cat([x['labels'] for x in outputs])).cpu().numpy()
        preds = gather(torch.cat([x['predicted_labels'] for x in outputs])).cpu().numpy()

        # optional embeddings
        def get_optional(key):
            if key in outputs[0]:
                return gather(torch.cat([x[key] for x in outputs])).cpu().numpy()
            return None

        v_appearance = get_optional("v_appearance")
        v_motion = get_optional("v_motion")
        s_motion = get_optional("s_motion")
        z_sensor = get_optional("z_sensor")
        z_video  = get_optional("z_video")
        bad      = get_optional("bad")

        # Hungarian mapping etc (기존 유지)
        num_clusters = self.clustering_manager.num_clusters
        remap = not self.mid_label

        if remap:
            labels_remapped = self._remap_pairwise_7(torch.tensor(labels)).numpy()
        else:
            labels_remapped = labels

        # ======== (1) Accuracy + Hungarian matching 수행 (이건 과제라 유지 가능) ========
        if self.local_rank == "0":

            raw_accuracy, new_mapping = compute_hungarian_matching(
                preds, labels_remapped, num_clusters
            )
            mapped_preds = np.array([new_mapping.get(int(c), int(c)) for c in preds])
            mapped_accuracy = (mapped_preds == labels_remapped).mean()

            wandb.log({"val_accuracy_raw": raw_accuracy})
            wandb.log({"val_accuracy_mapped": mapped_accuracy})

            # mapping 업데이트
            self.clustering_manager.mapping = new_mapping

            # ======== (2) Embedding 저장만 하기 (t-SNE 등 안함) ========
            save_dir = "/mnt/hdd4tb/junho/Opportunity++/tsne_cache"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"epoch_{self.epoch:04d}.npz")

            np.savez(
                save_path,
                features=features,
                labels=labels_remapped,
                preds=mapped_preds,
                v_motion=v_motion,
                s_motion=s_motion,
                z_sensor=z_sensor,
                z_video=z_video,
                v_appearance=v_appearance,
                bad=bad,
                num_clusters=num_clusters,
            )

            print(f"✔ Saved embeddings for epoch {self.epoch} → {save_path}")
        self.train()
        self.eval()
        features_gathered = gather(torch.cat([x['features'] for x in outputs]))
        labels_gathered = gather(torch.cat([x['labels'] for x in outputs]))
        predicted_labels_gathered = gather(torch.cat([x['predicted_labels'] for x in outputs]))
        print(f"evaluate_odc: Gathered {features_gathered.shape[0]} features from all ranks.")

        num_clusters = self.clustering_manager.num_clusters
        remap = not self.mid_label
        if remap:
            labels_remapped = self._remap_pairwise_7(labels_gathered)
        else:
            labels_remapped = labels_gathered

        if 'video_preds' in outputs[0]:
                video_preds_gathered = gather(torch.cat([x['video_preds'] for x in outputs]))
                video_labels_gathered = gather(torch.cat([x['labels'] for x in outputs]))
        else:
            video_preds_gathered = None

        # --- v_appearance 추가 ---
        if "v_appearance" in outputs[0]:
            v_appearance_gathered = gather(torch.cat([x["v_appearance"] for x in outputs]))
            v_appearance_np = v_appearance_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no v app key found in outputs")
            v_appearance_np = None

        # --- 새로 추가: v_motion 임베딩 ---
        if 'v_motion' in outputs[0]:
            v_motion_gathered = gather(torch.cat([x['v_motion'] for x in outputs]))
            v_motion_np = v_motion_gathered.cpu().numpy()
            print(f"evaluate: gathered v_motion {v_motion_gathered.shape}")
        else:
            v_motion_gathered = None
            print("evaluate: no v_motion in outputs")

        # --- [1️⃣ Bad 샘플 수집] ---
        if "bad" in outputs[0]:
            bad_gathered = gather(torch.cat([x["bad"] for x in outputs]))
            bad_np = bad_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no bad key found in outputs")
            bad_np = np.zeros(len(predicted_labels_gathered))

        # --- sensor_motion 추가 ---
        if "s_motion" in outputs[0]:
            s_motion_gathered = gather(torch.cat([x["s_motion"] for x in outputs]))
            s_motion_np = s_motion_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no s motion key found in outputs")
            s_motion_np = None
        
        # --- z_sensor 추가 ---
        if "z_sensor" in outputs[0]:
            z_sensor_gathered = gather(torch.cat([x["z_sensor"] for x in outputs]))
            z_sensor_np = z_sensor_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no z sensor key found in outputs")
            z_sensor_np = None
        
        # --- z_video 추가 ---
        if "z_video" in outputs[0]:
            z_video_gathered = gather(torch.cat([x["z_video"] for x in outputs]))
            z_video_np = z_video_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no z video key found in outputs")
            z_video_np = None

        # --- [2️⃣ Hungarian matching (기존 유지)] ---
        if self.local_rank == "0":
            # 🔹 추가: video classifier 평가
            if video_preds_gathered is not None:
                if remap:
                    video_labels_gathered = self._remap_pairwise_7(video_labels_gathered, num_clusters)
                video_preds_np = video_preds_gathered.cpu().numpy()
                video_labels_np = video_labels_gathered.cpu().numpy()

                mapping = getattr(self.clustering_manager, "mapping", None)
                if mapping is not None and len(mapping) > 0:
                    mapped_preds = np.array([mapping.get(int(p), int(p)) for p in video_preds_np])
                else:
                    mapped_preds = video_preds_np

                video_acc = np.mean(mapped_preds == video_labels_np)
                print(f"Video classifier accuracy (mapped): {video_acc:.4f}")
                wandb.log({"val/video_classifier_acc_mapped": video_acc})

            all_features = features_gathered.cpu().numpy()
            all_labels = labels_gathered.cpu().numpy()
            all_predicted_labels = predicted_labels_gathered.cpu().numpy()
            all_remapped_labels = labels_remapped.cpu().numpy()

            print(f"evaluate_odc: Calculating results on {len(all_features)} total samples.")
            print("Computing Hungarian matching...")

            raw_accuracy, new_mapping = compute_hungarian_matching(
                all_predicted_labels, all_remapped_labels, num_clusters
            )

            mapped_cluster_labels = np.array([new_mapping.get(c, c) for c in all_predicted_labels])
            mapped_accuracy = np.mean(mapped_cluster_labels == all_remapped_labels)
            print(f"Val Accuracy (Full Dataset): {mapped_accuracy:.4f}")

            if new_mapping is not None:
                self.clustering_manager.mapping = new_mapping
            print("new mapping", new_mapping)

            wandb.log({
                "val_accuracy_raw": raw_accuracy,
                "val_accuracy_mapped": mapped_accuracy
            })

            # --- [3️⃣ t-SNE 시각화 전용 라벨 수정] ---
            all_labels_tsne = all_remapped_labels.copy()
            # all_labels_tsne[bad_np == 1] = num_clusters          # bad → 새 class index
            mapped_cluster_labels_tsne = mapped_cluster_labels.copy()
            mapped_cluster_labels_tsne[bad_np == 1] = num_clusters  # pred도 동일하게 표시

            num_classes_for_tsne = num_clusters + 1
            bad_ratio = bad_np.mean() * 100

            # --- [4️⃣ t-SNE 시각화 호출] ---
            num_samples_for_tsne = min(1500, len(all_features))
            sample_indices = np.random.choice(len(all_features), num_samples_for_tsne, replace=False)

            try:
                print(f"Running t-SNE on {num_samples_for_tsne} samples (Bad {bad_ratio:.1f}%)...")
                fig_2d, fig_3d = visualize_tsne(
                    all_features[sample_indices],
                    all_labels_tsne[sample_indices],
                    mapped_cluster_labels[sample_indices],
                    prototypes=self.clustering_manager.centroids.detach().cpu().numpy(),
                    title=f"ODC Validation at Epoch {self.epoch} (Bad {bad_ratio:.1f}%)",
                    num_classes=num_clusters,
                    dataset_name=self.dataset_name,
                )
                wandb.log({
                    "val_tsne_2d": wandb.Image(fig_2d),
                    "val_tsne_3d": wandb.Image(fig_3d)
                })
                plt.close(fig_2d); plt.close(fig_3d)
            except Exception as e:
                print(f"Error during t-SNE visualization: {e}")

            # --- 🔥 추가: Motion t-SNE ---
            try:
                self.visualize_embedding(v_motion_np, all_labels, title="V_Motion")
            except Exception as e:
                print(f"Error during v motion t-SNE visualization: {e}")  
            # try:
            #     self.visualize_embedding(v_appearance_np, all_labels, title="V_Appearance")
            # except Exception as e:
            #     print(f"Error during v appearance t-SNE visualization: {e}")  
            try:
                self.visualize_embedding(s_motion_np, all_labels, title="S_Motion")
            except Exception as e:
                print(f"Error during s motion t-SNE visualization: {e}")  
            try:
                self.visualize_embedding(z_sensor_np, all_labels, title="Z_Sensor")
            except Exception as e:
                print(f"Error during z sensor t-SNE visualization: {e}")  
            try:
                self.visualize_embedding(z_video_np, all_labels, title="Z_Video")
            except Exception as e:
                print(f"Error during z video t-SNE visualization: {e}")  
            try:
                visualize_joint_space(z_video_np, z_sensor_np, all_labels, title="Z_video vs Z_sensor (Joint t-SNE)")
                wandb.log({"joint_z_tsne": wandb.Image(plt)})
                score = compute_alignment_score(z_video_np, z_sensor_np, all_labels)
                wandb.log({"alignment_score": score})
                cross_acc = cross_modal_retrieval(z_video_np, z_sensor_np)
                wandb.log({"cross_modal_retrieval": cross_acc})
            except Exception as e:
                print("Joint visualization error:", e)

              

        self.train()

    def visualize_embedding(self, some_np, all_labels, title="S_Motion"):
        num_samples_for_tsne = min(600, len(some_np))
        sample_indices = np.random.choice(len(some_np), num_samples_for_tsne, replace=False)

        print(f"Running {title} t-SNE on {num_samples_for_tsne} samples...")
        fig_motion_2d, fig_motion_3d = visualize_tsne(
            some_np[sample_indices],
            all_labels[sample_indices],  # 실제 라벨 기준
            None,   # ← pred_labels 대신 true_labels 전달
            prototypes=None,
            title=f"{title} t-SNE (Epoch {self.epoch})",
            num_classes=self.clustering_manager.num_clusters*2,
            dataset_name=self.dataset_name,
        )
        wandb.log({
            f"{title}_val_tsne_2d": wandb.Image(fig_motion_2d),
            f"{title}_val_tsne_3d": wandb.Image(fig_motion_3d),
        })
        plt.close(fig_motion_2d)
        plt.close(fig_motion_3d)

    def _remap_pairwise_7(self, labels_any):
        """
        0,2 -> 0 / 1,3 -> 1 / others -> // 2
        -1 (미할당)은 그대로 둠.
        """

        import numpy as np
        import torch

        if self.mid_label:
            return None

        if isinstance(labels_any, torch.Tensor):
            x = labels_any.clone().to(torch.long)
            device = x.device

            neg1_mask = (x == -1)
            m02 = (x == 0) | (x == 2)
            m13 = (x == 1) | (x == 3)
            others = ~(neg1_mask | m02 | m13)

            # 기본: 그대로
            y = x.clone()
            # 규칙 적용
            y = torch.where(m02, torch.zeros_like(y), y)
            y = torch.where(m13, torch.ones_like(y), y)
            y[others] = torch.div(x[others], 2, rounding_mode='floor')
            # -1 보존
            y[neg1_mask] = -1
            print(y)
            return y.to(device)

        else:  # numpy
            x = np.asarray(labels_any).astype(np.int64)
            y = x.copy()

            neg1_mask = (x == -1)
            m02 = (x == 0) | (x == 2)
            m13 = (x == 1) | (x == 3)
            others = ~(neg1_mask | m02 | m13)

            y[m02] = 0
            y[m13] = 1
            y[others] = x[others] // 2
            y[neg1_mask] = -1
            return y
        
    def _remap_pairwise_2(self, labels_any):
        """
        0,1 -> 0 / 2,3 -> 1 / others -> x % 2
        -1 (미할당)은 그대로 둠.
        """
        import numpy as np
        import torch

        if isinstance(labels_any, torch.Tensor):
            x = labels_any.clone().to(torch.long)
            device = x.device

            neg1_mask = (x == -1)
            m01 = (x == 0) | (x == 1)
            m23 = (x == 2) | (x == 3)
            others = ~(neg1_mask | m01 | m23)

            y = x.clone()
            y = torch.where(m01, torch.zeros_like(y), y)  # 0,1 -> 0
            y = torch.where(m23, torch.ones_like(y), y)   # 2,3 -> 1
            y[others] = x[others] % 2                     # others -> %2
            y[neg1_mask] = -1                             # -1 보존
            return y.to(device)

        else:  # numpy
            x = np.asarray(labels_any).astype(np.int64)
            y = x.copy()

            neg1_mask = (x == -1)
            m01 = (x == 0) | (x == 1)
            m23 = (x == 2) | (x == 3)
            others = ~(neg1_mask | m01 | m23)

            y[m01] = 0            # 0,1 -> 0
            y[m23] = 1            # 2,3 -> 1
            y[others] = x[others] % 2
            y[neg1_mask] = -1
            return y
