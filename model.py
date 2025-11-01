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
from PIL import Image, ImageDraw

from visualizes import visualize_tsne, visualize_sensor_name, START_INDEX, END_INDEX, visualize_cropped_tensor, denormalize, compute_hungarian_matching
from tqdm import tqdm
from method_utils import time_warp, match_target_to_recon, log_optical_flow_overlay_to_wandb


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


# class SensorModel(nn.Module):
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

class SensorModel(nn.Module):
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


#################################################################


class Clip4ClipVisionModel(nn.Module):
    
    def __init__(self):
        super().__init__() 

        # 1. CLIP 비전 모델 로드 (기존과 동일)
        self.video_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", attn_implementation="eager")
        
        # 'openai/clip-vit-base-patch32' 모델의 hidden_size는 768입니다.
        # 이 부분은 LoRA로 관리하지 않고 전체 파라미터를 학습(full-tuning)합니다.
        hidden_size = self.video_model.config.hidden_size # 768
        # ==========================================================================================
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
                "visual_projection"
            ],
            lora_dropout=0.05,
            bias="none"
        )
        
        self.video_model = get_peft_model(self.video_model, lora_config)
        # self.video_model.print_trainable_parameters()


    def forward(self, video: torch.Tensor, output_attentions: bool = False):
        if video.dim() == 4:
            video = video.unsqueeze(1)

        batch_size, n_frames, c, h, w = video.shape
        video_reshaped = video.view(batch_size * n_frames, c, h, w)

        # 💡 self.video_model 호출 시 output_attentions 인자 전달
        visual_output = self.video_model(
            pixel_values=video_reshaped,
            output_attentions=output_attentions
        )

        final_features = visual_output.last_hidden_state
        seq_len = final_features.shape[1]
        hidden_size = final_features.shape[2]
        final_features = final_features.view(batch_size, n_frames, seq_len, hidden_size)

        # 💡 output_attentions 값에 따라 반환값을 다르게 설정
        if output_attentions:
            # .attentions 값의 형태를 원래 비디오 차원에 맞게 복원
            # attentions는 튜플이므로 마지막 레이어의 어텐션만 사용
            attentions = visual_output.attentions[-1]
            num_heads = attentions.shape[1]
            attn_seq_len = attentions.shape[2]
            attentions = attentions.view(batch_size, n_frames, num_heads, attn_seq_len, attn_seq_len)
            
            return {
                "final_features": final_features,
                "attentions": attentions
            }
        else:
            return {
                "final_features": final_features
            }

# ---------------------------------------------------------------------
# Attention Head (Object branch saliency map)
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

# class MotionBranch(nn.Module):
#     """Temporal difference-based motion encoder."""
#     def __init__(self, in_channels=3, base_dim=32, latent_dim=256):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Conv3d(in_channels, base_dim, kernel_size=3, stride=(1,2,2), padding=1),
#             nn.BatchNorm3d(base_dim),
#             nn.ReLU(inplace=True),

#             nn.Conv3d(base_dim, base_dim * 2, kernel_size=3, stride=(1,2,2), padding=1),
#             nn.BatchNorm3d(base_dim * 2),
#             nn.ReLU(inplace=True),

#             nn.Conv3d(base_dim * 2, latent_dim, kernel_size=3, stride=(1,2,2), padding=1),
#             nn.BatchNorm3d(latent_dim),
#             nn.ReLU(inplace=True),
#         )
#         self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 시간축 유지
#         self.proj = nn.Linear(latent_dim, latent_dim)

#     def forward(self, video):  # [B, T, C, H, W]
#         x = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
#         feat = self.backbone(x)           # [B, D, T', H', W']
#         print("feat shape", feat.shape)
#         feat = self.spatial_pool(feat).squeeze(-1).squeeze(-1)  # [B, D, T']

#         # --- Temporal difference ---
#         motion_diff = feat[:, :, 1:] - feat[:, :, :-1]  # [B, D, T'-1]
#         v_motion = motion_diff.mean(dim=2)              # [B, D]
#         v_motion = self.proj(v_motion)                  # [B, D]

#         return v_motion

class MotionBranch(nn.Module):
    """Temporal difference + optical flow motion encoder."""
    def __init__(self, in_channels=5, base_dim=32, latent_dim=256):  # ✅ 3(RGB) + 2(Flow)
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(base_dim),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_dim, base_dim * 2, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(base_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_dim * 2, latent_dim, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, video, flows):

        # ✅ 시간 길이 맞춤
        if flows.shape[1] < video.shape[1]:
            pad = flows[:, -1:, :, :, :]
            flows = torch.cat([flows, pad], dim=1)

        # ✅ 채널 결합
        x = torch.cat([video, flows], dim=2)  # [B, T, 5, H, W]
        x = x.permute(0, 2, 1, 3, 4)          # [B, 5, T, H, W]

        feat = self.backbone(x)
        feat = self.spatial_pool(feat).squeeze(-1).squeeze(-1)
        motion_diff = feat[:, :, 1:] - feat[:, :, :-1]
        v_motion = motion_diff.mean(dim=2)
        v_motion = self.proj(v_motion)

        return v_motion

        return v_motion, diff_mag

# ---------------------------------------------------------------------
# Scene Branch (Global context)
# ---------------------------------------------------------------------
class SceneBranch(nn.Module):
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
# Object Branch (Local salient appearance)
# ---------------------------------------------------------------------
class ObjectBranch(nn.Module):
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
# VisionModel (MOSO-style Appearance Decomposition)
# ---------------------------------------------------------------------
class VisionModel(nn.Module):
    """
    MOSO + Motion branch (for Stage2 alignment)
    """
    def __init__(self, in_channels=3, base_dim=64, latent_dim=256):
        super().__init__()
        self.shared_encoder = SharedEncoder(in_channels, base_dim, out_dim=latent_dim)
        self.scene_branch = SceneBranch(in_dim=latent_dim, latent_dim=latent_dim)
        self.object_branch = ObjectBranch(in_dim=latent_dim, latent_dim=latent_dim)
        self.motion_branch = MotionBranch(in_channels=in_channels+2, base_dim=base_dim//2, latent_dim=latent_dim)

        # Appearance fusion (Scene + Object)
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, video, flows):
        """
        video: [B, T, C, H, W]
        returns dict with v_scene, v_object, v_appearance, v_motion
        """
        B, T, C, H, W = video.shape

        # 1️⃣ Shared 2D appearance encoding
        video_flat = video.view(B*T, C, H, W)
        shared_feat = self.shared_encoder(video_flat)
        _, D, Hf, Wf = shared_feat.shape
        shared_feat = shared_feat.view(B, T, D, Hf, Wf).mean(dim=1)

        v_scene = self.scene_branch(shared_feat)
        v_object = self.object_branch(shared_feat)

        fused = torch.cat([v_scene, v_object], dim=1)
        v_appearance = self.fuse(fused)  # [B, D]

        # 2️⃣ Motion branch (Stage2 전용)
        video_centered = video - video.mean(dim=[3, 4], keepdim=True)
        video_centered = video_centered / (video_centered.std(dim=[3, 4], keepdim=True) + 1e-6)

        # v_motion = self.motion_branch(video-video.mean(dim=1, keepdim=True))  # [B, D]
        v_motion = self.motion_branch(video_centered-video.mean(dim=1, keepdim=True), flows=flows)  # [B, D]

        return {
            "v_scene": v_scene,
            "v_object": v_object,
            "v_appearance": v_appearance,
            "v_motion": v_motion,
            "vis_v": video_centered-video.mean(dim=1, keepdim=True)
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

    def _gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all replicas into a single tensor."""
        # 현재 분산 그룹의 GPU 개수를 가져옵니다.
        world_size = dist.get_world_size()
        if world_size == 1:
            print("world_size == 1, no gather needed")
            return tensor

        # 입력 텐서가 반드시 GPU에 있도록 보장합니다.

        # 2. ⭐️ 입력 텐서를 현재 프로세스의 올바른 GPU로 이동시킵니다.
        # 이렇게 하면 rank 1은 cuda:1로, rank 2는 cuda:2로 텐서를 옮깁니다.
        
        tensor = tensor.cuda()
        # 1. 최종적으로 모일 전체 텐서의 크기를 계산하고, '같은 device'에 빈 텐서를 생성합니다.
        shape = (world_size * tensor.shape[0], *tensor.shape[1:])
        gathered_tensor = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        
        # 2. all_gather_into_tensor를 호출하여 빈 텐서를 채웁니다.
        dist.all_gather_into_tensor(gathered_tensor, tensor)
        return gathered_tensor

    def update_samples_memory(self, idx: torch.Tensor,
                              feature: torch.Tensor):
        """Update samples memory."""
        assert self.initialized
        # print(f"[{self.local_rank}] Updating samples memory for {idx.shape[0]} samples.")
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize

        idx = self._gather(idx)
        feature_norm = self._gather(feature_norm)
        
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
        print("update_samples_memory", change_ratio)
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
        """클러스터 크기에 근거한 클래스 가중치를 계산합니다."""
        # 클러스터 크기가 0인 경우를 방지하기 위한 정규화
        histogram = np.bincount(
            self.label_bank.cpu().numpy(), minlength=self.num_clusters)
        cluster_size = torch.tensor(histogram, device=self.centroids.device)
        print("Cluster sizes:", cluster_size)
        normalized_sizes = cluster_size + 1e-8
        
        # 클러스터 크기의 그대로의 역수를 가중치로 사용
        # 더 강한 가중치를 적용하기 위해 거듭제곱을 높임
        weights = 1.0 / torch.pow(normalized_sizes, self.class_weight_power)
        
        # 최소 가중치와 최대 가중치 간의 차이가 너무 크지 않도록 조정
        # 작은 클러스터에 더 높은 가중치를 주되, 그 차이가 너무 크지 않도록 함
        max_weight = weights.max()
        min_weight = weights.min()
        if max_weight > min_weight * 10:
            # 최대 가중치가 최소의 10배를 넘지 않도록 조절
            weights = torch.clamp(weights, min=max_weight/10)
        
        # 가중치 정규화 (0~1 범위 밖으로 팬는 것 방지)
        weights = weights / weights.sum() * self.num_clusters
        return weights

# --- 4. Clustering 모델 ---
class ClusteringModel(nn.Module):
    def __init__(self, encoder, embedding_dim, num_sensors, num_clusters, datamodule, top_k=1, prototype_cache_dir="./cache", dataset_name="custom_dataset", min_cluster_size=30):
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
        # # --- ★★★ 추가된 코드: 전체 학습 데이터 t-SNE 시각화 ★★★ ---
        # if rank_str == "0": # Rank 0에서만 시각화 수행
        #     print(f"[{rank}] Generating initial t-SNE plot from the *entire* train dataset...")
        #     print(f"[{rank}] WARNING: This might take a long time!")
            
        #     all_features_list = []
        #     all_labels_list = []
            
        #     # try:
        #         # 데이터 로더 전체 순회하며 특징 및 레이블 수집 (CPU 사용)
        #         # (GPU 메모리 부족 방지 위해 CPU 사용 후 t-SNE 시 필요하면 샘플링)
        #     print(f"[{rank}] Collecting features and labels from train_dataloader...")
        #     for batch in tqdm(self.train_dataloader, desc=f"[{rank}] Collecting Features"):
        #         videos, sensors, labels, sample_ids = batch
                
        #         sensors = sensors.to(device) 
                
        #         # 모델 forward 호출 (특징 추출)
        #         _, _, features = self(sensors, return_features=True) 
        #         # features = self.projection_layer(features) # 필요시
                
        #         all_features_list.append(features.detach().cpu())
        #         all_labels_list.append(labels.detach().cpu()) # 레이블도 CPU로

        #     # 리스트를 하나의 텐서/배열로 합치기
        #     all_features = torch.cat(all_features_list).numpy()
        #     all_labels = torch.cat(all_labels_list).numpy()
            
        #     print(f"[{rank}] Collected {len(all_features)} total samples.")
            
        #     # --- ★★★ 레이블 병합 로직 추가 ★★★ ---
        #     labels_to_plot = all_labels
        #     # print(f"[{rank}] Merging labels...")
        #     # merged_labels = np.zeros_like(all_labels) # 결과를 저장할 새 배열
        #     # for i, l in enumerate(all_labels):
        #     #     if l == 0 or l == 1:
        #     #         merged_labels[i] = 0
        #     #     elif l == 2 or l == 3:
        #     #         merged_labels[i] = 1
        #     #     else:
        #     #         # 정수 나눗셈 // 사용
        #     #         merged_labels[i] = l % 2 
            
        #     # labels_to_plot = merged_labels # 시각화에는 병합된 레이블 사용
        #     print(f"[{rank}] Labels merged.")
        #     # --- ★★★ 레이블 병합 끝 ★★★ ---

        #     # (선택) 레이블 병합 로직 (필요하다면 여기에 적용)
        #     # labels_to_plot = all_labels 

        #     # 샘플링 (데이터가 너무 많을 경우)
        #     num_samples_for_tsne = min(20000, len(all_features)) # 샘플 수 증가 (시간 더 걸림)
        #     if num_samples_for_tsne < len(all_features):
        #         print(f"[{rank}] Sampling {num_samples_for_tsne} for t-SNE...")
        #         sample_indices = np.random.choice(len(all_features), num_samples_for_tsne, replace=False)
        #         features_subset = all_features[sample_indices]
        #         labels_subset = labels_to_plot[sample_indices]
        #     else:
        #         features_subset = all_features
        #         labels_subset = labels_to_plot

        #     # t-SNE 실행
        #     perplexity_value = min(30, len(features_subset) - 1)
        #     if perplexity_value <= 0: perplexity_value = 1.0
        #     from sklearn.manifold import TSNE
        #     print(f"[{rank}] Running t-SNE (2D) on {len(features_subset)} samples...")
        #     tsne_2d = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, metric="cosine")
        #     reduced_features_2d = tsne_2d.fit_transform(features_subset)

        #     print(f"[{rank}] Running t-SNE (3D) on {len(features_subset)} samples...")
        #     tsne_3d = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, metric="cosine")
        #     reduced_features_3d = tsne_3d.fit_transform(features_subset)

        #     # 시각화 (간단 버전)
        #     unique_labels = np.unique(labels_subset)
        #     num_unique_labels = len(unique_labels)
        #     if num_unique_labels <= 20: cmap = plt.cm.get_cmap('tab20', num_unique_labels) 
        #     else: cmap = plt.cm.get_cmap('viridis', num_unique_labels)
        #     colors = cmap(np.linspace(0, 1, num_unique_labels))
        #     label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        #     label_names = {label: f"Class_{label}" for label in unique_labels} # 임시 이름

        #     # 2D
        #     fig_2d = plt.figure(figsize=(10, 8)); ax_2d = fig_2d.add_subplot(111); handles = []
        #     for i in unique_labels:
        #         mask = (labels_subset == i); color=label_to_color[i]; label_name=label_names[i]
        #         ax_2d.scatter(reduced_features_2d[mask, 0], reduced_features_2d[mask, 1], color=color, label=label_name, alpha=0.7)
        #         if not any(h.get_label() == label_name for h in handles): handles.append(plt.Line2D([],[],color=color, marker='o', ls='', ms=8, label=label_name))
        #     ax_2d.set_title("Initial t-SNE (Full Train Set - 2D)"); ax_2d.legend(handles=handles, loc='best')
            
        #     # 3D
        #     fig_3d = plt.figure(figsize=(10, 8)); ax_3d = fig_3d.add_subplot(111, projection='3d'); handles_3d = []
        #     for i in unique_labels:
        #         mask = (labels_subset == i); color=label_to_color[i]; label_name=label_names[i]
        #         ax_3d.scatter(reduced_features_3d[mask, 0], reduced_features_3d[mask, 1], reduced_features_3d[mask, 2], color=color, label=label_name, alpha=0.7)
        #         if not any(h.get_label() == label_name for h in handles_3d): handles_3d.append(plt.Line2D([],[],color=color, marker='o', ls='', ms=8, label=label_name))
        #     ax_3d.set_title("Initial t-SNE (Full Train Set - 3D)"); ax_3d.legend(handles=handles_3d, loc='best')

        #     # WandB 로깅
        #     wandb.log({
        #         "initial_train_tsne_2d": wandb.Image(fig_2d),
        #         "initial_train_tsne_3d": wandb.Image(fig_3d)
        #     })
        #     plt.close(fig_2d); plt.close(fig_3d) 
        #     print(f"[{rank}] Initial Train t-SNE logged to WandB.")

        #     # except Exception as e:
        #     #     print(f"[{rank}] Error during initial full train t-SNE visualization: {e}")
        # # --- ★★★ 추가된 코드 끝 ★★★ ---
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
            
            for i, (videos, sensors, labels, sample_ids) in enumerate(self.train_dataloader):
                # print("initializing prototypes - processing batch", i, "on rank", rank)
                sensors = sensors.to(device)
                idx, _ = sample_ids
                idx = idx.clone().to(dtype=torch.long, device=device)
                
                _, features = self(sensors, return_features=True)
                # ... (get_representative_sensor_feature 및 projection_layer 로직) ...
                
                local_features.append(features) # GPU 상태로 유지
                local_idx.append(idx) # GPU 상태로 유지
                print(f"[{rank}] Processed batch {i+1}/{len(self.train_dataloader)}")
            
            local_features_tensor = torch.cat(local_features, dim=0)
            local_idx_tensor = torch.cat(local_idx, dim=0)

            # --- 2-3. All-Gather로 전체 특징과 ID 복제 ---
            all_features_gpu = self.clustering_manager._gather(local_features_tensor)
            all_idx_gpu = self.clustering_manager._gather(local_idx_tensor)
            
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

    def forward(self, x, idx=None, return_features=False, labels=None, step="train"):
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
        features_gathered = self.clustering_manager._gather(torch.cat([x['features'] for x in outputs]))
        labels_gathered = self.clustering_manager._gather(torch.cat([x['labels'] for x in outputs]))
        predicted_labels_gathered = self.clustering_manager._gather(torch.cat([x['predicted_labels'] for x in outputs]))
        print(f"evaluate_odc: Gathered {features_gathered.shape[0]} features from all ranks.")

        num_clusters = self.clustering_manager.num_clusters
        remap=True
        if remap:
            labels_remapped = self._remap_pairwise_7(labels_gathered, num_clusters)
        else:
            labels_remapped = labels_gathered

        if 'video_preds' in outputs[0]:
                video_preds_gathered = self.clustering_manager._gather(torch.cat([x['video_preds'] for x in outputs]))
                video_labels_gathered = self.clustering_manager._gather(torch.cat([x['labels'] for x in outputs]))
        else:
            video_preds_gathered = None

        # --- 새로 추가: v_motion 임베딩 ---
        if 'v_motion' in outputs[0]:
            v_motion_gathered = self.clustering_manager._gather(torch.cat([x['v_motion'] for x in outputs]))
            v_motion_np = v_motion_gathered.cpu().numpy()
            print(f"evaluate: gathered v_motion {v_motion_gathered.shape}")
        else:
            v_motion_gathered = None
            print("evaluate: no v_motion in outputs")

        # --- [1️⃣ Bad 샘플 수집] ---
        if "bad" in outputs[0]:
            bad_gathered = self.clustering_manager._gather(torch.cat([x["bad"] for x in outputs]))
            bad_np = bad_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no bad key found in outputs")
            bad_np = np.zeros(len(predicted_labels_gathered))

        # --- sensor_motion 추가 ---
        if "s_motion" in outputs[0]:
            s_motion_gathered = self.clustering_manager._gather(torch.cat([x["s_motion"] for x in outputs]))
            s_motion_np = s_motion_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no s motion key found in outputs")
            s_motion_np = None
        
        # --- z_sensor 추가 ---
        if "z_sensor" in outputs[0]:
            z_sensor_gathered = self.clustering_manager._gather(torch.cat([x["z_sensor"] for x in outputs]))
            z_sensor_np = z_sensor_gathered.cpu().numpy()
        else:
            print("⚠️ warning: no z sensor key found in outputs")
            z_sensor_np = None
        
        # --- z_video 추가 ---
        if "z_video" in outputs[0]:
            z_video_gathered = self.clustering_manager._gather(torch.cat([x["z_video"] for x in outputs]))
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
            num_samples_for_tsne = min(70000, len(all_features))
            sample_indices = np.random.choice(len(all_features), num_samples_for_tsne, replace=False)

            try:
                print(f"Running t-SNE on {num_samples_for_tsne} samples (Bad {bad_ratio:.1f}%)...")
                fig_2d, fig_3d = visualize_tsne(
                    all_features[sample_indices],
                    all_labels[sample_indices],
                    mapped_cluster_labels[sample_indices],
                    prototypes=self.clustering_manager.centroids.detach().cpu().numpy(),
                    title=f"ODC Validation at Epoch {self.epoch} (Bad {bad_ratio:.1f}%)",
                    num_classes=num_clusters*2,
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
              

        self.train()

    def visualize_embedding(self, some_np, all_labels, title="S_Motion"):
        num_samples_for_tsne = min(1500, len(some_np))
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
            f"{title}_val_tsne_motion_2d": wandb.Image(fig_motion_2d),
            f"{title}_val_tsne_motion_3d": wandb.Image(fig_motion_3d),
        })
        plt.close(fig_motion_2d)
        plt.close(fig_motion_3d)

    def _remap_pairwise_7(self, labels_any, num_clusters):
        """
        0,2 -> 0 / 1,3 -> 1 / others -> // 2
        -1 (미할당)은 그대로 둠.
        """

        import numpy as np
        import torch

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
            print(y)
            return y
