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
from method_utils import time_warp


#################################################################

import torch
import torch.nn as nn

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_type="max", embedding_size=32):
        super().__init__()
        if pool_type == "max":
            pool_fn = torch.nn.MaxPool1d(kernel_size=3)
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
    
class SensorModel(nn.Module):
    def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.GroupNorm(1, sensor_channels),
            Block(sensor_channels, input_dim, 5),
            Block(input_dim, input_dim *2, 3),
            Block(input_dim *2, input_dim *2, 3, pool_type="adaptive", embedding_size=32),
            torch.nn.GroupNorm(4, input_dim *2),
            torch.nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )
        self.ssl_head = torch.nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = torch.nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch):
        emb = self.backbone(batch)[1][0] # Last hidden state
        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)
        out = {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}
        return out

# class SensorModel(nn.Module):
#     """각 센서 채널을 독립적으로 처리한 후, 그 특징들을 GRU로 융합하는 모델"""
#     def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
#         super().__init__()
#         self.in_channels = sensor_channels
#         self.per_channel_dim = input_dim

#         # 1. 각 채널에 독립적으로 적용될 작은 1D CNN
#         # 모든 채널이 이 동일한 CNN을 공유함
#         self.channel_encoder = nn.Sequential(
#             nn.Conv1d(1, 8, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(8, input_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1) # 각 채널의 시계열을 하나의 벡터로
#         )
        
#         # 2. 채널별 특징들을 융합(fusion)하기 위한 GRU
#         self.fusion_gru = nn.GRU(
#             input_size=input_dim,
#             hidden_size=size_embeddings,
#             batch_first=True
#         )

#     def forward(self, x):
#         # x shape: (Batch, Channels, SequenceLength)
#         B, C, L = x.shape
        
#         # 1. 각 채널을 독립적으로 처리하기 위해 차원 변경
#         # (B, C, L) -> (B * C, 1, L)
#         x_reshaped = x.view(-1, 1, L)
        
#         # 2. 채널별 인코딩
#         channel_features = self.channel_encoder(x_reshaped) # -> (B * C, per_channel_dim, 1)
#         channel_features = channel_features.squeeze(-1) # -> (B * C, per_channel_dim)
        
#         # 3. 다시 배치 형태로 복원
#         # (B * C, per_channel_dim) -> (B, C, per_channel_dim)
#         channel_features_batched = channel_features.view(B, C, self.per_channel_dim)
        
#         # (여기서 top_k 센서 선택 로직을 적용할 수 있습니다)
#         # 예를 들어, 특정 규칙으로 k개의 채널 인덱스를 선택하여
#         # selected_features = channel_features_batched[:, top_k_indices, :] 와 같이 처리한 후 fusion_gru에 넣을 수 있습니다.
        
#         # 4. GRU로 채널 간의 관계를 학습하여 최종 특징 추출
#         _, hidden = self.fusion_gru(channel_features_batched)
        
#         out = {"emb": hidden[-1]} # -> (B, feature_dim)
#         return out
    
# # 1. 채널 어텐션 (Squeeze-and-Excitation) 블록 정의
# # 이 부분은 수정 없이 그대로 사용합니다.
# class SEBlock(nn.Module):
#     """
#     Squeeze-and-Excitation 블록으로, 채널별 중요도를 동적으로 학습합니다.
#     """
#     def __init__(self, num_channels, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         # Squeeze 과정: 글로벌 정보를 요약
#         self.squeeze = nn.AdaptiveAvgPool1d(1)
#         # Excitation 과정: 어떤 채널이 중요한지 학습
#         self.excitation = nn.Sequential(
#             nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         batch_size, channels, _ = x.shape
#         # Squeeze를 통해 (batch, channels, 1) -> (batch, channels)로 변환
#         y = self.squeeze(x).view(batch_size, channels)
#         # Excitation을 통해 채널별 중요도(가중치) 계산
#         y = self.excitation(y).view(batch_size, channels, 1)
#         # 원래의 입력(x)에 중요도를 곱하여 스케일 조정 (Rescale)
#         return x * y.expand_as(x)

# # 2. 양방향 풀링이 적용된 최종 센서 인코더
# class SensorModel(nn.Module):
#     """
#     순간적인 행동(positive & negative peaks)을 포착하기 위해 
#     양방향 풀링(Bi-directional Pooling)과 채널 어텐션을 사용하는 센서 인코더.
#     """
#     def __init__(self, sensor_channels, size_embeddings=128):
#         super(SensorModel, self).__init__()
        
#         # 1D CNN 레이어
#         self.conv1 = nn.Conv1d(in_channels=sensor_channels, out_channels=64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(128)

#         # 채널 어텐션 블록
#         self.se_block = SEBlock(num_channels=128)
        
#         # Max Pooling 레이어 (양방향 풀링에 공통으로 사용)
#         self.temporal_pool = nn.AdaptiveMaxPool1d(1)
        
#         # 최종 임베딩을 위한 MLP
#         # --- 수정된 부분 --- #
#         # 양방향 풀링으로 max와 min 특징이 결합되므로, 입력 차원이 2배가 됨 (128 -> 256)
#         self.fc = nn.Sequential(
#             nn.Linear(128 * 2, 128), # 입력 차원 수정
#             nn.ReLU(inplace=True),
#             nn.Linear(128, size_embeddings)
#         )

#     def forward(self, x):
#         # 입력 데이터 shape: (batch_size, num_channels, sequence_length)
        
#         # CNN으로 특징 추출
#         x = F.relu(self.bn1(self.conv1(x)))
#         features = F.relu(self.bn2(self.conv2(x)))
        
#         # 채널 어텐션 적용
#         features = self.se_block(features)
        
#         # --- 양방향 풀링 (Bi-directional Pooling) --- #
#         # 1. 양의 방향으로 가장 큰 순간 포착
#         max_pool_features = self.temporal_pool(features) 
        
#         # 2. 음의 방향으로 가장 큰 순간 포착 (min pooling 효과)
#         # features에 -를 붙여서 max_pool을 하면 가장 작은 값을 찾는 효과
#         min_pool_features = self.temporal_pool(-features)
        
#         # 3. 두 특징을 채널 차원에서 결합 (concatenate)
#         # 결합 전에 min_pool_features에 다시 -를 붙여 원래 값으로 복원
#         pooled_features = torch.cat([max_pool_features, -min_pool_features], dim=1)
        
#         # Flatten
#         pooled_features = pooled_features.view(pooled_features.size(0), -1) 
        
#         # MLP로 최종 임베딩 생성
#         embedding = self.fc(pooled_features)
        
#         return {"emb": embedding}

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

#################################################################


class DinoVisionModel(nn.Module):
    """
    DINO로 사전 학습된 ViT를 특징 추출기로 사용하는 클래스 (수정본).
    """
    def __init__(self): # num_classes는 특징 추출만 하므로 필요 없음
        super().__init__()

        # DINO ViT 모델 로드
        self.video_model = AutoModel.from_pretrained("facebook/dinov2-small")

        # 2. ViTModel 아키텍처에 맞는 LoRA 설정
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "attention.attention.query",
                "attention.attention.key",
                "attention.attention.value",
                "attention.output.dense",
                "intermediate.dense",
                "output.dense",
            ],
            lora_dropout=0.05,
            bias="none"
        )

        self.video_model = get_peft_model(self.video_model, lora_config)
        self.video_model.print_trainable_parameters() # 학습 가능한 파라미터 수 확인


    def forward(self, video: torch.Tensor):
        if video.dim() == 4:
            video = video.unsqueeze(1)

        batch_size, n_frames, c, h, w = video.shape
        video_reshaped = video.view(batch_size * n_frames, c, h, w)

        # 1. output_hidden_states=True 옵션 없이 모델 호출
        visual_output = self.video_model(video_reshaped)

        # 2. .hidden_states 대신 .last_hidden_state를 직접 사용
        final_features = visual_output.last_hidden_state

        # 최종 특징 텐서의 형태를 원래 비디오 차원에 맞게 복원
        seq_len = final_features.shape[1]
        hidden_size = final_features.shape[2]
        final_features = final_features.view(batch_size, n_frames, seq_len, hidden_size)

        # 최종 특징만 반환하도록 수정
        return {
            "final_features": final_features
        }


#################################################################


class LocalisationNetwork(nn.Module):
    """
    F_A를 입력받아 어파인 변환 행렬 theta를 회귀하는 작은 CNN.
    """
    def __init__(self, input_channels: int, patch_grid_size: int):
        super().__init__()
        
        # F_A를 처리하기 위한 CNN 구조
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # H, W -> H/2, W/2
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) # H/2, W/2 -> H/4, W/4
        )

        # CNN 출력 크기를 동적으로 계산
        final_grid_size = patch_grid_size // 4
        final_channels = 64
        flattened_size = final_channels * final_grid_size * final_grid_size

        # 어파인 변환 행렬 theta (2x3)의 6개 파라미터를 회귀
        self.regressor = nn.Linear(flattened_size, 6)

        # 학습 안정성을 위해 항등 변환(identity transform)으로 초기화
        self.regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, features_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_2d (torch.Tensor): (B*T, C, H_patch, W_patch) 형태의 특징 맵
        Returns:
            torch.Tensor: (B*T, 2, 3) 형태의 어파인 변환 행렬 theta
        """
        x = self.cnn(features_2d)
        x = x.reshape(x.size(0), -1)
        theta = self.regressor(x)
        theta = theta.view(-1, 2, 3) # (B*T, 2, 3) 형태로 변환
        return theta


#################################################################


class AttentionBridge(nn.Module):
    """
    LocalisationNetwork와 STN을 통합하여 어텐션 브릿지 역할을 수행.
    (수정 버전: 시간 축으로 평균화된 대표 특징을 사용하여 클립 전체에 적용될 단일 변환 행렬을 계산)
    """

    def __init__(self, input_hidden_size: int, patch_grid_size: int, target_size: tuple = (96, 96)):
        super().__init__()
        self.localisation_net = LocalisationNetwork(input_hidden_size, patch_grid_size)
        self.target_size = target_size # V'의 목표 해상도 (H_t, W_t)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        (2단계 수정 버전) 특징 맵(F_A)을 직접 변환하여 변환된 특징 맵(F'_A)을 생성합니다.
        Args:
            features (torch.Tensor): (B, T, Seq_Len, Hidden_Size) 형태의 ViT 특징 (F_A)
        Returns:
            torch.Tensor: (B, T, Hidden_Size, H_t, W_t) 형태의 변환된 특징 맵 클립 (F'_A)
        """
        batch_size, n_frames, seq_len, hidden_size = features.shape

        # --- 1. 대표 특징 생성 (Temporal Average Pooling) ---
        # 시간 축 평균을 통해 단일 변환 행렬 계산에 사용할 안정적인 특징을 만듭니다.
        # (B, T, Seq_Len, Hidden_Size) -> (B, Seq_Len, Hidden_Size)
        robust_features = torch.mean(features, dim=1)

        # --- 2. LocalisationNetwork 입력 준비 ---
        # CLS 토큰을 제외하고 2D 그리드 형태로 변환합니다.
        patch_features = robust_features[:, 1:, :]
        num_patches = patch_features.shape[1]
        patch_grid_h = patch_grid_w = int(num_patches ** 0.5)
        patch_features_2d = patch_features.permute(0, 2, 1).view(
            batch_size, hidden_size, patch_grid_h, patch_grid_w
        )

        # --- 3. LocalisationNetwork를 통해 단일 통합 theta 계산 ---
        # (B, Hidden_Size, H_patch, W_patch) -> (B, 2, 3)
        theta = self.localisation_net(patch_features_2d)

        # --- 4. STN을 이용해 F'_A 생성 (Feature-level Cropping) ---
        # 변환할 대상인 원본 특징맵(features)을 2D 그리드 형태로 준비합니다.
        # CLS 토큰을 제외하고 (B*T, Hidden_Size, H_patch, W_patch) 형태로 변환합니다.
        patch_features_all_frames = features[:, :, 1:, :].reshape(batch_size * n_frames, seq_len - 1, hidden_size)
        features_to_transform = patch_features_all_frames.permute(0, 2, 1).view(
            batch_size * n_frames, hidden_size, patch_grid_h, patch_grid_w
        )

        # 단일 theta를 모든 프레임에 적용하기 위해 T 차원으로 반복합니다.
        theta_repeated = repeat(theta, 'b c h -> (b t) c h', t=n_frames)

        # grid_sample에 사용할 목표 크기를 지정합니다.
        grid_target_size = torch.Size([batch_size * n_frames, hidden_size, self.target_size[0], self.target_size[1]])

        # 반복된 theta를 이용해 샘플링 그리드를 생성합니다.
        grid = F.affine_grid(theta_repeated, grid_target_size, align_corners=False)

        # **원본 특징 맵**과 그리드를 이용해 변환된 특징 맵을 샘플링합니다.
        transformed_features = F.grid_sample(features_to_transform, grid, align_corners=False, padding_mode="border")

        # --- 5. 최종 출력 형태 복원 ---
        # (B*T, D, H_t, W_t) -> (B, T, D, H_t, W_t)
        transformed_features = transformed_features.view(batch_size, n_frames, hidden_size, self.target_size[0], self.target_size[1])

        return transformed_features
            

#################################################################


class Conv2Plus1D(nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: tuple, 
                stride: tuple, 
                padding: tuple,
                mid_channels: int = None):
        """
        (2+1)D 컨볼루션 블록. 3D 컨볼루션을 2D 공간과 1D 시간 컨볼루션으로 분해합니다.

        Args:
            in_channels (int): 입력 채널의 수.
            out_channels (int): 출력 채널의 수.
            kernel_size (tuple): (temporal, height, width) 형태의 커널 크기 튜플.
            stride (tuple): (temporal, height, width) 형태의 스트라이드 튜플.
            padding (tuple): (temporal, height, width) 형태의 패딩 튜플.
            mid_channels (int, optional): 공간 컨볼루션과 시간 컨볼루션 사이의 중간 채널 수.
            None이면 out_channels와 동일하게 설정됩니다.
        """
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # 공간 컨볼루션 (2D)
        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)

        # 시간 컨볼루션 (1D)
        self.temporal_conv = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서. Shape: (B, C, T, H, W)
        
        Returns:
            torch.Tensor: 출력 텐서.
        """
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.temporal_conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


#################################################################


class AppearanceEncoder(nn.Module):
    """
    비디오 특징을 인코딩하여 외형 벡터(v_appearance)를 추출합니다.
    (2+1)D 컨볼루션 스택과 시간 평균 풀링(temporal average pooling)을 사용합니다.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): 입력 채널의 수 (특징 추출기로부터).
            mid_channels (int): 중간 레이어의 채널 수.
            out_channels (int): 최종 외형 벡터의 크기.
        """
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # [수정됨] kernel_size, stride, padding을 튜플로 전달
            Conv2Plus1D(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),  # 시간(T) stride=1, 공간(H,W) stride=2
                padding=(1, 1, 1)
            ),
            Conv2Plus1D(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),  # 모든 차원에서 stride=1
                padding=(1, 1, 1)
            )
        )

        # 공간 차원을 풀링하여 채널당 하나의 특징만 남깁니다.
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, T, C, H, W) 형태의 입력 텐서.

        Returns:
            torch.Tensor: (B, out_channels) 형태의 외형 벡터 v_appearance.
        """

        # (2+1)D 컨볼루션 블록을 통과시킵니다.
        x = self.conv_blocks(x)

        # 시간 평균 풀링 (Temporal Average Pooling)
        # x shape: (B, C_out, T_out, H_out, W_out) -> (B, C_out, H_out, W_out)
        x = x.mean(dim=2)

        # 공간 풀링 및 flatten
        # x shape: (B, C_out, H_out, W_out) -> (B, C_out, 1, 1)
        x = self.spatial_pool(x)

        # x shape: (B, C_out, 1, 1) -> (B, C_out)
        v_appearance = torch.flatten(x, 1)

        return v_appearance
    
    
#################################################################


class MotionEncoder(nn.Module):
    """
    비디오 특징을 인코딩하여 동작 벡터(v_motion)를 추출합니다.
    (2+1)D 컨볼루션 스택과 GRU를 사용하여 시간적 역학을 포착합니다.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, rnn_hidden_size: int):
        """
        Args:
            in_channels (int): 입력 채널의 수.
            mid_channels (int): 중간 컨볼루션 레이어의 채널 수.
            out_channels (int): 컨볼루션 블록의 출력 채널 수. 이는 RNN의 입력 크기가 됩니다.
            rnn_hidden_size (int): GRU의 은닉 상태 크기. 최종 v_motion 벡터의 차원이 됩니다.
        """
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # [수정됨] kernel_size, stride, padding을 튜플로 전달
            Conv2Plus1D(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1)
            ),
            Conv2Plus1D(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1)
            )
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.rnn = nn.GRU(
            input_size=out_channels,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, T, C, H, W) 형태의 입력 텐서.
        Returns:
            torch.Tensor: (B, rnn_hidden_size) 형태의 동작 벡터 v_motion.
        """
        b, c, t, h, w = x.shape 

        x = self.conv_blocks(x)
        _, c_out, t_out, h_out, w_out = x.shape

        # 공간 풀링을 위해 텐서 reshape
        x = x.permute(0, 2, 1, 3, 4)      # -> (B, T_out, C_out, H_out, W_out)
        x = x.reshape(b * t_out, c_out, h_out, w_out)
        
        # 공간 풀링 적용
        x = self.spatial_pool(x)          # -> (B * T_out, C_out, 1, 1)
        x = torch.flatten(x, 1)           # -> (B * T_out, C_out)
        
        # RNN 입력을 위해 시퀀스 형태로 복원
        x = x.view(b, t_out, c_out)       # -> (B, T_out, C_out)
        
        # GRU 통과
        _, h_n = self.rnn(x)              # h_n shape: (1, B, rnn_hidden_size)
        
        # 최종 v_motion 벡터 추출
        v_motion = h_n.squeeze(0)         # -> (B, rnn_hidden_size)

        return v_motion
    

#################################################################


class VisionModel(nn.Module):
    def __init__(self, image_size: int, target_size: tuple = (96, 96)):
        super().__init__()

        # --- 1단계: 특징 추출 및 ROI 지역화 ---
        self.feature_extractor = Clip4ClipVisionModel() 
       
        path_load_pretrained_clip4clip = "/home/jaemo/Multimodal/PRIMUS/saved/i2c/clip_finetuned_opportunity_50_weakly_supervised.pth"
        self.feature_extractor.load_state_dict(torch.load(path_load_pretrained_clip4clip))
        hidden_size = self.feature_extractor.video_model.config.hidden_size  # e.g., 384

        self.patch_size = self.feature_extractor.video_model.config.patch_size
        patch_grid_size = image_size // self.patch_size

        self.crop_size = min(image_size, 224)  # CLIP 모델의 입력 크기와 동일하게 설정

        self.attention_bridge = AttentionBridge(
            input_hidden_size=hidden_size,
            patch_grid_size=patch_grid_size,
            target_size=target_size
        )

        # --- 2단계: Appearance & Motion 인코딩 ---
        # 인코더들의 채널 크기를 정의합니다.
        encoder_mid_channels = 512
        appearance_out_channels = 256
        motion_out_channels = 256 # RNN의 입력 크기가 됩니다.
        motion_rnn_hidden_size = 256

        self.appearance_encoder = AppearanceEncoder(
            in_channels=hidden_size,
            mid_channels=encoder_mid_channels,
            out_channels=appearance_out_channels
        )

        self.motion_encoder = MotionEncoder(
            in_channels=hidden_size,
            mid_channels=encoder_mid_channels,
            out_channels=motion_out_channels,
            rnn_hidden_size=motion_rnn_hidden_size
        )

        self.local_rank = os.environ.get("LOCAL_RANK", "0")

        # feature_extractor의 출력 차원을 받아서 num_classes로 매핑하는 분류 헤드
        feature_dim = self.feature_extractor.video_model.config.hidden_size # e.g., 768
        self.classifier = nn.Linear(feature_dim, 7)
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo_model.classes = [0]  # 0번 클래스가 'person' 입니다.
        
    # forward 함수 예시
    def forward(self, video_batch):
        # (B, T, C, H, W) -> (B*T, C, H, W)
        B, T, C, H, W = video_batch.shape
        video_reshaped = video_batch.view(B*T, C, H, W)
        
        features_output = self.feature_extractor.video_model(pixel_values=video_reshaped)
        
        # [CLS] 토큰 특징 사용 (첫 번째 토큰)
        cls_features = features_output.last_hidden_state[:, 0] # (B*T, hidden_size)
        
        # 시간 축으로 평균내어 비디오 전체 특징 계산
        video_features = cls_features.view(B, T, -1).mean(dim=1) # (B, hidden_size)
        
        # 분류 헤드를 통과시켜 로짓 계산
        logits = self.classifier(video_features)
        
        return {'logits': logits}
        
    # YourMainModel 클래스 내의 patch_selection 메서드
    def patch_selection(self, videos_batch: torch.Tensor, labels) -> torch.Tensor:
    # 1. 입력 텐서 정보 저장
        # 입력 가정: (B, T, C, H_orig, W_orig) -> (B, T, C, 480, 640)
        B, T, C, H_orig, W_orig = videos_batch.shape

        print(f"[Rank {self.local_rank}] Patch Selection Input Shape: {videos_batch.shape}")
        
        # 모델이 요구하는 입력 크기
        MODEL_INPUT_SIZE = (224, 224)
        
        # (B, T, C, H, W) -> (B*T, C, H, W)
        video_reshaped_orig = videos_batch.view(B * T, C, H_orig, W_orig)

        # 2. Step 1: 저해상도에서 단서 찾기
        # 원본 해상도 프레임들을 모델 입력 크기로 리사이즈
        video_resized_for_model = resize(video_reshaped_orig, size=MODEL_INPUT_SIZE)
        
        # 리사이즈된 이미지로 어텐션 맵 계산
        with torch.no_grad():
            model_output = self.feature_extractor.video_model(
                pixel_values=video_resized_for_model,
                output_attentions=True
            )
        
        # 어텐션 맵을 다시 비디오 단위로 재구성
        attentions = model_output.attentions[-1]
        _ , n_heads, seq_len, _ = attentions.shape
        attentions = attentions.view(B, T, n_heads, seq_len, seq_len)

        cropped_videos_list = []
        iou_scores_list = [] # 🌟 IoU 점수를 저장할 리스트 추가

        # for i in range(B): 루프 전체를 이 코드로 교체해주세요.

        # for i in range(B): 루프 전체를 이 코드로 교체해주세요.

        for i in range(B):
            # --- 1. 어텐션으로 Crop 영역 좌표(top, left) 계산 ---
            # (이 부분은 기존과 동일하므로 생략)
            attentions_per_video = attentions[i]
            cls_attentions = attentions_per_video[:, :, 0, 1:]
            attention_weights = cls_attentions.mean(dim=[0, 1])
            h_patches = MODEL_INPUT_SIZE[0] // self.patch_size
            w_patches = MODEL_INPUT_SIZE[1] // self.patch_size
            attention_map_2d = attention_weights.reshape(h_patches, w_patches)
            max_idx_flat = torch.argmax(attention_map_2d)
            max_idx_y_lowres = (max_idx_flat // w_patches).item()
            max_idx_x_lowres = (max_idx_flat % w_patches).item()
            center_y_lowres = (max_idx_y_lowres + 0.5) * self.patch_size
            center_x_lowres = (max_idx_x_lowres + 0.5) * self.patch_size
            scale_h, scale_w = H_orig / MODEL_INPUT_SIZE[0], W_orig / MODEL_INPUT_SIZE[1]
            center_y_highres = int(center_y_lowres * scale_h)
            center_x_highres = int(center_x_lowres * scale_w)
            top = max(0, min(center_y_highres - self.crop_size // 2, H_orig - self.crop_size))
            left = max(0, min(center_x_highres - self.crop_size // 2, W_orig - self.crop_size))

            # --- 2. YOLO 및 시각화 준비 ---
            gt_boxes = []
            person_found_score = 0.0

            # 2-1. 원본 프레임 역정규화 및 Crop (0~1 float)

            # with torch.no_grad(): 블록 전체를 이 코드로 교체해주세요.

            # with torch.no_grad() 블록 전체를 교체

            with torch.no_grad():
                # 2-1. 원본 프레임 역정규화 및 Crop (0~1 float)
                original_frame_tensor = denormalize(videos_batch[i])[T // 2]
                cropped_frame_float_0_1 = original_frame_tensor[:, top:top + self.crop_size, left:left + self.crop_size]
                h_cropped_orig, w_cropped_orig = cropped_frame_float_0_1.shape[1:]

                # 2-2. Crop 이미지를 YOLO 입력 크기로 리사이즈
                YOLO_INPUT_SIZE = 640
                resized_for_yolo_float_0_1 = resize(cropped_frame_float_0_1.unsqueeze(0), size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)).squeeze(0)
                input_tensor_for_yolo = resized_for_yolo_float_0_1.mul(255).byte().unsqueeze(0)

                # --- 3. YOLO 실행 및 후처리 ---
                yolo_results_list = self.yolo_model(input_tensor_for_yolo)
                
                # 새로운 후처리 함수는 너비/높이 인자가 필요 없습니다.
                final_outputs = self.postprocess_yolo_output(yolo_results_list[0])
                
                # normalized_boxes는 0~1 범위의 '비율' 좌표를 가집니다.
                normalized_boxes = final_outputs[0]
                
                if normalized_boxes.shape[0] > 0:
                    person_found_score = 1.0
                    # print("[DEBUG] Normalized Boxes:", normalized_boxes)
                    # print("[DEBUG] Cropped Frame Size (HxW):", h_cropped_orig, w_cropped_orig)

                    
                    # 🌟🌟🌟 단 한번의, 최종 스케일링! 🌟🌟🌟
                    # '비율' 좌표를 -> '픽셀' 좌표로 변환합니다.
                    pixel_boxes = normalized_boxes.clone() # 복사해서 사용
                    pixel_boxes[:, [0, 2]] *= w_cropped_orig # x 좌표에 너비 곱하기
                    pixel_boxes[:, [1, 3]] *= h_cropped_orig # y 좌표에 높이 곱하기

                    # 이제 gt_boxes는 올바른 픽셀 좌표를 가집니다.
                    gt_boxes = pixel_boxes[:, :4].cpu().tolist()

            iou_scores_list.append(person_found_score)

            # --- 4. 시각화 ---
            if self.local_rank == "0":
                frame_np_for_drawing = cropped_frame_float_0_1.permute(1, 2, 0).mul(255).byte().cpu().numpy()
                dummy_crop_box = [0, 0, w_cropped_orig - 1, h_cropped_orig - 1]
                
                debug_image = self.draw_boxes_on_frame(
                    frame_np=frame_np_for_drawing,
                    yolo_boxes=gt_boxes,
                    crop_box=dummy_crop_box
                )

                success_str = "✅ Found" if person_found_score > 0 else "❌ Not_Found"
                log_title = f"{success_str}_Label_{labels[i]}"
                wandb.log({log_title: wandb.Image(debug_image, file_type="png")})

            # --- 5. 다음 모델로 전달할 Crop 비디오 준비 ---
            single_video_orig = videos_batch[i].permute(1, 0, 2, 3)
            cropped_video_for_model = crop(single_video_orig, top, left, self.crop_size, self.crop_size)
            cropped_video_restored = cropped_video_for_model.permute(1, 0, 2, 3)
            cropped_videos_list.append(cropped_video_restored)

        # --- 함수 마지막 부분 ... ---

        # --- 함수 마지막 (기존과 동일) ---
        final_cropped_batch = torch.stack(cropped_videos_list, dim=0)
        iou_scores = torch.tensor(iou_scores_list, device=videos_batch.device)
        return final_cropped_batch, iou_scores
    # YourMainModel 클래스 내부

    @staticmethod
    def postprocess_yolo_output(prediction, conf_thres=0.1, iou_thres=0.45):
        """
        YOLO의 원시 출력을 후처리하여 "정규화된(0~1) 좌표"를 가진
        최종 바운딩 박스를 반환합니다.
        """
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        
        for xi, x in enumerate(prediction):  # 배치 내 각 이미지에 대해 처리
            
            # 🌟🌟🌟 바로 이 한 줄이 모든 것을 해결합니다! 🌟🌟🌟
            # Logit을 0~1 사이의 비율/확률 값으로 변환합니다.
            x=x.sigmoid()

            # 이제 0~1로 변환된 값을 기준으로 신뢰도 필터링을 수행합니다.
            x = x[x[..., 4] > conf_thres]
            
            if not x.shape[0]:
                continue
            
            # 클래스 점수 계산
            box = x[:, :4] # 이제 box는 0~1 사이의 cx,cy,w,h 입니다.
            x[:, 5:] *= x[:, 4:5]
            
            # 박스 좌표를 (cx,cy,w,h) -> (x1,y1,x2,y2)로 변환 (스케일링 없음)
            box_normalized = torch.empty_like(box)
            box_normalized[:, 0] = box[:, 0] - box[:, 2] / 2
            box_normalized[:, 1] = box[:, 1] - box[:, 3] / 2
            box_normalized[:, 2] = box[:, 0] + box[:, 2] / 2
            box_normalized[:, 3] = box[:, 1] + box[:, 3] / 2
            
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box_normalized, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            x = x[x[:, 5] == 0] # 'person' 클래스 필터링
            
            if not x.shape[0]:
                continue

            # NMS 적용
            boxes, scores = x[:, :4], x[:, 4]
            nms_indices = torchvision.ops.nms(boxes, scores, iou_thres)
            output[xi] = x[nms_indices]
            
        return output

    # from PIL import Image, ImageDraw, ImageFont # 파일 상단에 추가해주세요.
# import cv2 # 이 함수에서는 더 이상 cv2를 사용하지 않습니다.

    @staticmethod
    def draw_boxes_on_frame(frame_np, yolo_boxes, crop_box):
        """
        Pillow(PIL)를 사용하여 원본 프레임에 YOLO 박스와 Crop 박스를 그립니다.
        """
        img = Image.fromarray(frame_np)
        draw = ImageDraw.Draw(img)
        print(f"Drawing {len(yolo_boxes)} YOLO boxes and crop box {crop_box}")
        
        # YOLO 박스 그리기 (초록색)
        for box in yolo_boxes:
            x1, y1, x2, y2 = box

            # 🌟🌟🌟 좌표 강제 보정 (안전장치) 🌟🌟🌟
            # x1이 x2보다 크거나, y1이 y2보다 큰 '뒤집힌' 박스를 방지합니다.
            # 두 x좌표 중 작은 값을 x1으로, 큰 값을 x2로 강제 지정합니다.
            corrected_x1 = min(x1, x2)
            corrected_y1 = min(y1, y2)
            corrected_x2 = max(x1, x2)
            corrected_y2 = max(y1, y2)
            print("Corrected Box:", corrected_x1, corrected_y1, corrected_x2, corrected_y2)

            # 보정된 좌표로 박스를 그립니다.
            draw.rectangle(
                [(corrected_x1, corrected_y1), (corrected_x2, corrected_y2)], 
                outline="green", 
                width=3
            )
            draw.text((corrected_x1, corrected_y1 - 10), "YOLO", fill="green")

        # Crop 영역 테두리 그리기 (빨간색)
        draw.rectangle(crop_box, outline="red", width=3)
        
        return img # Pillow 이미지 객체를 그대로 반환
    
    def calculate_iou(self, boxA, boxesB):
        """한 개의 박스(boxA)와 여러 개의 박스(boxesB) 사이의 IoU를 계산합니다."""
        # boxA: [x1, y1, x2, y2]
        # boxesB: torch.Tensor of shape (N, 4)
        
        xA = torch.max(boxA[0], boxesB[:, 0])
        yA = torch.max(boxA[1], boxesB[:, 1])
        xB = torch.min(boxA[2], boxesB[:, 2])
        yB = torch.min(boxA[3], boxesB[:, 3])

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxesB[:, 2] - boxesB[:, 0]) * (boxesB[:, 3] - boxesB[:, 1])
        
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou
 

#################################################################


# --- 3. 메모리 뱅크 관리자 ---
class ClusteringManager(nn.Module):
    def __init__(self, num_clusters, feature_dim, momentum=0.9, temperature=0.1, device='cuda', local_rank=0):
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
        self.min_cluster_size = 30  # 이 값보다 작으면 비어있다고 간주
        
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
        print(f"[{self.local_rank}] Updating samples memory for {idx.shape[0]} samples.")
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
    def _partition_max_cluster(
            self, max_cluster: np.ndarray):
        """Partition the largest cluster into two sub-clusters."""
        assert self.local_rank == "0"
        max_cluster_idx = np.where(self.label_bank.cpu().numpy() == max_cluster)[0]

        assert len(max_cluster_idx) >= 2
        max_cluster_features = self.feature_bank[max_cluster_idx, :]
        if np.any(np.isnan(max_cluster_features.cpu().numpy())):
            raise Exception('Has nan in features.')
        kmeans_ret = self.kmeans.fit(max_cluster_features.cpu())
        sub_cluster1_idx = max_cluster_idx[kmeans_ret.labels_ == 0]
        sub_cluster2_idx = max_cluster_idx[kmeans_ret.labels_ == 1]
        if not (len(sub_cluster1_idx) > 0 and len(sub_cluster2_idx) > 0):
            print(
                'Warning: kmeans partition fails, resort to random partition.')
            sub_cluster1_idx = np.random.choice(
                max_cluster_idx, len(max_cluster_idx) // 2, replace=False)
            sub_cluster2_idx = np.setdiff1d(
                max_cluster_idx, sub_cluster1_idx, assume_unique=True)
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
    def __init__(self, encoder, embedding_dim, num_sensors, num_clusters, train_dataloader, top_k=1, prototype_cache_dir="./cache", dataset_name="custom_dataset"):
        super().__init__()
        # 딥러닝 백본 선택
        self.encoder = encoder
        self.top_k = top_k
        self.local_rank = os.environ.get("LOCAL_RANK", "0")
        # Clustering 관리자
        self.clustering_manager = ClusteringManager(num_clusters=num_clusters, feature_dim=embedding_dim, local_rank=self.local_rank)
        self.projection_layer = nn.Linear(num_sensors, embedding_dim)
        self.epoch = 0
        self.train_dataloader = train_dataloader
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
            local_sample_ids = []
            
            for i, (videos, sensors, labels, sample_ids) in enumerate(self.train_dataloader):
                # print("initializing prototypes - processing batch", i, "on rank", rank)
                sensors = sensors.to(device)
                sample_ids = sample_ids.clone().to(dtype=torch.long, device=device)
                
                _, features = self(sensors, return_features=True)
                # ... (get_representative_sensor_feature 및 projection_layer 로직) ...
                
                local_features.append(features) # GPU 상태로 유지
                local_sample_ids.append(sample_ids) # GPU 상태로 유지
                print(f"[{rank}] Processed batch {i+1}/{len(self.train_dataloader)}")
            
            local_features_tensor = torch.cat(local_features, dim=0)
            local_ids_tensor = torch.cat(local_sample_ids, dim=0)

            # --- 2-3. All-Gather로 전체 특징과 ID 복제 ---
            all_features_gpu = self.clustering_manager._gather(local_features_tensor)
            all_ids_gpu = self.clustering_manager._gather(local_ids_tensor)
            
            # --- 3. Rank 0에서만 K-Means 실행 및 초기화 ---
            if rank_str == "0":
                print(f"[{rank}] Running KMeans...")
                
                # all_features_cpu = all_features_gpu.cpu()
                all_ids_cpu = all_ids_gpu.cpu()
                
                num_total_samples = len(self.train_dataloader.dataset)
                feature_bank_temp = torch.empty(num_total_samples, self.clustering_manager.feature_dim, device="gpu")
                feature_bank_temp[all_ids_cpu, ...] = all_features_gpu # ID를 사용한 최종 재배치
                
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

    def forward(self, x, sample_ids=None, return_features=False, labels=None, step="train"):
        features = self.encoder(x)["emb"]
        if step == "train" or step == "val":

            # 클러스터 유사도 점수 계산
            representative_feature = self.get_representative_sensor_feature(x, labels, num_total_sensors=END_INDEX-START_INDEX+1, top_k=self.top_k, id=sample_ids)
            representative_feature = self.projection_layer(representative_feature)
            # alpha = self.gate(features)
            alpha=1
            # if self.local_rank == "0" and labels is not None:
                # wandb.l   og("alpha", alpha, labels)
            features = features + alpha * representative_feature
            
        similarity_scores = self.clustering_manager.compute_similarity_scores(features)
        # similarity_scores = self.cls_head(features)

        if return_features:
            return similarity_scores, features
        return similarity_scores
    
    @torch.no_grad()
    def get_pseudo_labels(self, sample_ids):
        """(개선된 버전) 메모리 뱅크에서 pseudo label을 효율적으로 조회합니다."""
        # sample_ids는 Dataset에서 온 정수 인덱스의 '리스트'라고 가정
        
        # 1. label_bank가 있는 디바이스 정보를 가져옵니다.
        device = self.clustering_manager.label_bank.device
        
        # 2. 파이썬 리스트를 모델과 같은 디바이스의 텐서로 변환합니다.
        print(sample_ids)
        sids_tensor = sample_ids.detach().clone().to(dtype=torch.long, device=device)
        
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
        
        # 각 채널(센서)의 분산 계산 (max - min)
        ranges = torch.quantile(torch.abs(imu_batch), q=0.99, dim=2)
        # ranges = torch.max(torch.abs(imu_batch), dim=2).values
        # ranges = torch.mean(torch.abs(imu_batch), dim=2)

        # ranges = torch.var(imu_batch, dim=2)
        # 가장 분산이 큰 센서의 인덱스 찾기 (분산이 0인것 제외)
        return ranges
    
        min_range, _ = torch.min(ranges, dim=1, keepdim=True)
        max_range, _ = torch.max(ranges, dim=1, keepdim=True)
        weighted_features = (ranges - min_range) / (max_range - min_range + 1e-8)
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
        return final_rule_feature
    
    def augment_imu_data(self, imu_data):
        return time_warp(imu_data)
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        
    @torch.no_grad()
    def evaluate(self, outputs):
        """
        미리 train step에서 계산된 outputs를 사용하여 평가를 수행합니다 (validation epoch 끝에서 호출하는 것보다 성능 향상이 더딜 수 있음).
        """

        self.eval()
        features_gathered = self.clustering_manager._gather(torch.cat([x['features'] for x in outputs]))
        labels_gathered = self.clustering_manager._gather(torch.cat([x['labels'] for x in outputs]))
        predicted_labels_gathered = self.clustering_manager._gather(torch.cat([x['predicted_labels'] for x in outputs]))
        print(f"evaluate_odc: Gathered {features_gathered.shape[0]} features from all ranks.")

        if self.local_rank == "0":
            # 1. 텐서를 NumPy 배열로 변환
            all_features = features_gathered.cpu().numpy()
            all_labels = labels_gathered.cpu().numpy()
            all_predicted_labels = predicted_labels_gathered.cpu().numpy()
        
            print(f"evaluate_odc: Calculating results on {len(all_features)} total samples.")

            # 2. 정확도 계산 (헝가리안 매칭)
            print("Computing Hungarian matching...")
            raw_accuracy, new_mapping = compute_hungarian_matching(
                all_predicted_labels, all_labels, self.clustering_manager.num_clusters
            )

            mapped_cluster_labels = np.array([new_mapping.get(c, c) for c in all_predicted_labels])
            mapped_accuracy = np.mean(mapped_cluster_labels == all_labels)
            print(f"Val Accuracy (Full Dataset): {mapped_accuracy:.4f}")
                  
            if new_mapping is not None:
                self.clustering_manager.mapping = new_mapping
                # mapping은 rank 0 (evaluate)에서만 사용됨

            # 3. 로깅 (전달받은 LightningModule의 logger 사용)
            wandb.log({
                "val_accuracy_raw": raw_accuracy,
                "val_accuracy_mapped": mapped_accuracy
            })
            
            # 4. 시각화 (t-SNE)
            # CPU 과부하 방지를 위해 샘플링 적용
            num_samples_for_tsne = min(2000, len(all_features))
            sample_indices = np.random.choice(len(all_features), num_samples_for_tsne, replace=False)
            
            try:
                print(f"Running t-SNE on a subset of {num_samples_for_tsne} samples...")
                fig_2d, fig_3d = visualize_tsne(
                    all_features[sample_indices], 
                    all_labels[sample_indices], 
                    mapped_cluster_labels[sample_indices],
                    prototypes=self.clustering_manager.centroids.detach().cpu().numpy(),
                    title=f"ODC Validation at Epoch {self.epoch}",
                    num_classes=self.clustering_manager.num_clusters,
                    dataset_name=self.dataset_name
                )
                wandb.log({
                    "val_tsne_2d": wandb.Image(fig_2d),
                    "val_tsne_3d": wandb.Image(fig_3d)
                })
                plt.close(fig_2d); plt.close(fig_3d)
            except Exception as e:
                print(f"Error during t-SNE visualization: {e}")

        self.train()