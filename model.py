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

from visualization import visualize_tsne, visualize_sensor_name, START_INDEX, END_INDEX
from utils import compute_hungarian_matching
from tqdm import tqdm


#################################################################


class Block(nn.Module):
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
            pool_fn,
        )
        
    def forward(self, batch):
        return self.net(batch)
    

# class SensorModel(nn.Module):
#     def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
#         super().__init__()

#         num_groups_for_input = 1

#         self.backbone = nn.Sequential(
#             nn.GroupNorm(num_groups_for_input, sensor_channels),
#             Block(sensor_channels, input_dim, 10),
#             Block(input_dim, input_dim, 5),
#             Block(input_dim, input_dim, 5, pool_type="adaptive", embedding_size=32),
#             nn.GroupNorm(4, input_dim),
#             nn.GRU(
#                 batch_first=True, input_size=input_dim, hidden_size=size_embeddings
#             ),
#         )

#     def forward(self, batch):
#         # GRU는 (output, h_n) 형태의 튜플을 반환합니다.
#         # h_n (마지막 타임스텝의 은닉 상태)을 사용합니다.
#         # h_n의 shape: (num_layers, batch_size, hidden_size)
#         _, last_hidden_state = self.backbone(batch)
        
#         # GRU 레이어가 하나이므로 첫 번째 요소를 선택하고, batch 차원을 유지하기 위해 squeeze(0) 대신 [0]을 사용합니다.
#         emb = last_hidden_state[0] # Shape: (batch_size, hidden_size)        
        
#         out = {"emb": emb}
#         return out


class SensorModel(nn.Module):
    """각 센서 채널을 독립적으로 처리한 후, 그 특징들을 GRU로 융합하는 모델"""
    def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.in_channels = sensor_channels
        self.per_channel_dim = input_dim

        # 1. 각 채널에 독립적으로 적용될 작은 1D CNN
        # 모든 채널이 이 동일한 CNN을 공유함
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 각 채널의 시계열을 하나의 벡터로
        )
        
        # 2. 채널별 특징들을 융합(fusion)하기 위한 GRU
        self.fusion_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=size_embeddings,
            batch_first=True
        )

    def forward(self, x):
        # x shape: (Batch, Channels, SequenceLength)
        B, C, L = x.shape
        
        # 1. 각 채널을 독립적으로 처리하기 위해 차원 변경
        # (B, C, L) -> (B * C, 1, L)
        x_reshaped = x.view(-1, 1, L)
        
        # 2. 채널별 인코딩
        channel_features = self.channel_encoder(x_reshaped) # -> (B * C, per_channel_dim, 1)
        channel_features = channel_features.squeeze(-1) # -> (B * C, per_channel_dim)
        
        # 3. 다시 배치 형태로 복원
        # (B * C, per_channel_dim) -> (B, C, per_channel_dim)
        channel_features_batched = channel_features.view(B, C, self.per_channel_dim)
        
        # (여기서 top_k 센서 선택 로직을 적용할 수 있습니다)
        # 예를 들어, 특정 규칙으로 k개의 채널 인덱스를 선택하여
        # selected_features = channel_features_batched[:, top_k_indices, :] 와 같이 처리한 후 fusion_gru에 넣을 수 있습니다.
        
        # 4. GRU로 채널 간의 관계를 학습하여 최종 특징 추출
        _, hidden = self.fusion_gru(channel_features_batched)
        
        out = {"emb": hidden[-1]} # -> (B, feature_dim)
        return out
    
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


    def forward(self, video: torch.Tensor):
        if video.dim() == 4:
            video = video.unsqueeze(1)

        batch_size, n_frames, c, h, w = video.shape

        # 비디오의 각 프레임(T개)을 배치 차원(B)으로 펼쳐, 모든 프레임을 독립적으로 처리하도록 만듭니다. 
        # 이는 '시간 축 평균 연산을 수행하기 전'의 정보를 모두 유지하는 핵심적인 부분
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
        hidden_size = self.feature_extractor.video_model.config.hidden_size  # e.g., 384

        patch_size = self.feature_extractor.video_model.config.patch_size
        patch_grid_size = image_size // patch_size

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

    def forward(self, video: torch.Tensor) -> dict:
        features_dict = self.feature_extractor(video)

        # 1. 특징 추출 (F_A)
        final_features = features_dict["final_features"]

        # 2. Attention Bridge를 통해 변환된 특징 맵 (F'_A) 생성
        # transformed_features shape: (B, T, D, H_t, W_t)
        transformed_features = self.attention_bridge(final_features)

        transformed_features = transformed_features.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)

        # 3. Appearance 및 Motion 벡터 추출
        v_appearance = self.appearance_encoder(transformed_features) # [B, 256]
        v_motion = self.motion_encoder(transformed_features) # [B, 256]

        # --- 최종 출력 통합 ---
        final_output = {
            "v_appearance": v_appearance,
            "v_motion": v_motion,
            "transformed_features": transformed_features,
            "final_features": final_features
        }

        return final_output


#################################################################


# --- 3. 메모리 뱅크 관리자 ---
class ClusteringManager(nn.Module):
    def __init__(self, num_clusters, feature_dim, momentum=0.01, temperature=0.1, device='cuda', local_rank=0):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.local_rank = local_rank

        if self.local_rank == 0:
            # key: sample_id, value: feature
            self.feature_bank = torch.zeros((10000, feature_dim),
                                            dtype=torch.float32)
            # key: sample_id, value: cluster_id
            self.label_bank = None
        
        # 클러스터 중심점(centroids) 초기화 - 더 넓게 분포되도록 초기화
        # 각 차원마다 균등 분포를 사용하여 더 잘 분산되도록 함
        centroids = torch.rand(num_clusters, feature_dim, device=device) * 2.0 - 1.0  # [-1, 1] 범위의 균등 분포
        
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
        self.min_cluster_size = 10  # 이 값보다 작으면 비어있다고 간주
        
        # 가중치 계산을 위한 상수
        self.class_weight_power = 0.5  # 클러스터 크기에 적용할 거듭제곱

        self.mapping = None

        self.kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
        # 전체 데이터셋의 pseudo label을 계산하고 메모리 뱅크에 저장하는 함수

    @torch.no_grad()
    def _compute_centroids_idx(self, cinds):
        """Compute a few centroids."""
        assert self.local_rank == 0
        num = len(cinds)
        centroids = torch.zeros((num, self.feature_dim), dtype=torch.float32)
        for i, c in enumerate(cinds):
            idx = np.where(self.label_bank.numpy() == c)[0]
            centroids[i, :] = self.feature_bank[idx, :].mean(dim=0)
        return centroids

    def _compute_centroids(self):
        """Compute all non-empty centroids."""
        assert self.local_rank == 0
        label_bank_np = self.label_bank.numpy()
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
            return tensor

        # 입력 텐서가 반드시 GPU에 있도록 보장합니다.
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
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize

        idx = self._gather(idx)
        feature_norm = self._gather(feature_norm)
        
        idx = idx.cpu()
        if self.local_rank == 0:
            feature_old = self.feature_bank[idx, ...].cuda()
            feature_new = (1 - self.momentum) * feature_old + \
                self.momentum * feature_norm
            feature_norm = feature_new / (
                feature_new.norm(dim=1).view(-1, 1) + 1e-10)
            self.feature_bank[idx, ...] = feature_norm.cpu()
        dist.barrier()
        dist.broadcast(feature_norm, 0)
        # compute new labels
        # similarity_to_centroids = self.compute_similarity_scores(feature_norm.permute(1, 0))
        feature_norm = feature_norm.permute(1, 0)
        centroids_norm = F.normalize(self.centroids, dim=1)
        similarity_to_centroids = torch.mm(centroids_norm,
                                           feature_norm)  # CxN
        newlabel = similarity_to_centroids.argmax(dim=0)  # cuda tensor
        newlabel_cpu = newlabel.cpu()
        change_ratio = (newlabel_cpu != self.label_bank[idx]
                        ).sum().float().cuda() / float(newlabel_cpu.shape[0])
        self.label_bank[idx] = newlabel_cpu.clone()  # copy to cpu
        print("update_samples_memory", change_ratio)
        return change_ratio

    @torch.no_grad()
    def update_centroids_memory(self, cinds = None):
        """Update centroids memory."""
        if self.local_rank == 0:
            if cinds is None:
                center = self._compute_centroids()
                self.centroids.copy_(center)
            else:
                center = self._compute_centroids_idx(cinds)
                self.centroids[
                    torch.LongTensor(cinds).cuda(), :] = center.cuda()
        dist.broadcast(self.centroids, 0)

        print("update_centroids_memory", self.centroids.shape)

    @torch.no_grad()
    def deal_with_small_clusters(self):
        """
        Gather all label_banks, perform clustering logic on rank 0,
        and return the updated centroids.
        """
        # 1. 모든 GPU의 label_bank를 rank 0으로 모읍니다.
        #    _gather 함수는 모든 GPU에서 호출되어야 합니다.
        global_label_bank = self._gather(self.label_bank)

        # 2. Rank 0 에서만 모든 계산을 수행합니다.
        if self.local_rank == 0:
            # 글로벌 데이터를 기준으로 small_clusters를 안전하게 계산
            global_histogram = np.bincount(
                global_label_bank.cpu().numpy(), minlength=self.num_clusters)
            small_clusters = np.where(global_histogram < self.min_cluster_size)[0].tolist()

            if len(small_clusters) == 0:
                # 변경 사항이 없으면 현재 centroids를 그대로 반환
                return self.centroids

            print(f'[Rank 0] Dealing with {len(small_clusters)} small clusters.')

            # 재할당 로직 수행 (모든 데이터가 Rank 0에 있으므로 동기화 불필요)
            for s in small_clusters:
                idx = np.where(global_label_bank.cpu().numpy() == s)[0]
                if len(idx) == 0:
                    continue
                
                # feature_bank도 모든 GPU에 걸쳐 동일한 복사본이 있어야 합니다.
                # (만약 아니라면 feature_bank도 gather가 필요합니다)
                inclusion = np.setdiff1d(np.arange(self.num_clusters), np.array(small_clusters), assume_unique=True)
                inclusion_tensor = torch.from_numpy(inclusion).cuda()

                # feature_bank에서 idx에 해당하는 부분만 가져와야 합니다.
                # feature_bank가 분산되어 있다면, 이 부분도 수정이 필요합니다.
                # 여기서는 self.feature_bank가 모든 GPU에 복제되어 있다고 가정합니다.
                gathered_features = self._gather(self.feature_bank) # 예시: feature_bank도 gather

                target_idx = torch.mm(
                    self.centroids[inclusion_tensor, :],
                    gathered_features[idx, :].cuda().permute(1, 0)
                ).argmax(dim=0)
                
                target = inclusion_tensor[target_idx]
                global_label_bank[idx] = target.cpu()

            # 모든 재할당 후, Rank 0에서 최종 centroids 계산
            # (이 로직은 _compute_centroids 같은 함수를 호출해야 할 수 있습니다)
            final_center = self._compute_centroids(global_label_bank, gathered_features) # 예시
            self.centroids.copy_(final_center)
        dist.broadcast(self.centroids, 0)
        # 3. Rank 0은 계산된 centroids를 반환, 나머지는 현재 자신의 centroids를 반환
        return self.centroids

    @torch.no_grad()
    def _partition_max_cluster(
            self, max_cluster: np.ndarray):
        """Partition the largest cluster into two sub-clusters."""
        assert self.local_rank == 0
        max_cluster_idx = np.where(self.label_bank == max_cluster)[0]

        assert len(max_cluster_idx) >= 2
        max_cluster_features = self.feature_bank[max_cluster_idx, :]
        if np.any(np.isnan(max_cluster_features.numpy())):
            raise Exception('Has nan in features.')
        kmeans_ret = self.kmeans.fit(max_cluster_features)
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
        print("empty_clusters", empty_clusters)
        for e in empty_clusters:
            assert (self.label_bank != e).all().item(), \
                f'Cluster #{e} is not an empty cluster.'
            
            # 1. 마스터에서만 max_cluster를 계산
            if self.local_rank == 0:
                max_cluster_val = np.bincount(
                    self.label_bank, minlength=self.num_clusters).argmax().item()
                # 값을 담을 텐서 생성
                max_cluster_tensor = torch.tensor([max_cluster_val], dtype=torch.int64).cuda()
            else:
                # 다른 프로세스들은 값을 받을 빈 텐서 생성
                max_cluster_tensor = torch.zeros(1, dtype=torch.int64).cuda()
            
            # 2. 모든 프로세스에 max_cluster 값을 broadcast
            dist.broadcast(max_cluster_tensor, 0)
            
            # 3. 이제 모든 프로세스가 동일한 max_cluster 값을 가짐
            max_cluster = max_cluster_tensor.item()
            # gather partitioning indices
            if self.local_rank == 0:
                sub_cluster1_idx, sub_cluster2_idx = \
                    self._partition_max_cluster(max_cluster)
                if len(sub_cluster1_idx) == 0 or len(sub_cluster2_idx) == 0:
                    print(f"Warning: empty partition at cluster {max_cluster}")
                    continue
                size1 = torch.LongTensor([len(sub_cluster1_idx)]).cuda()
                size2 = torch.LongTensor([len(sub_cluster2_idx)]).cuda()
                sub_cluster1_idx_tensor = torch.from_numpy(
                    sub_cluster1_idx).long().cuda()
                sub_cluster2_idx_tensor = torch.from_numpy(
                    sub_cluster2_idx).long().cuda()
            else:
                size1 = torch.LongTensor([0]).cuda()
                size2 = torch.LongTensor([0]).cuda()
            print("all_reduce ", self.local_rank)
            dist.all_reduce(size1)
            print("get reduce 1")
            dist.all_reduce(size2)
            print("get sizes", size1, size2)
            if self.local_rank != 0:
                sub_cluster1_idx_tensor = torch.zeros(
                    (size1.item(), ), dtype=torch.int64).cuda()
                sub_cluster2_idx_tensor = torch.zeros(
                    (size2.item(), ), dtype=torch.int64).cuda()
            dist.broadcast(sub_cluster1_idx_tensor, 0)
            dist.broadcast(sub_cluster2_idx_tensor, 0)

            if self.local_rank != 0:
                sub_cluster1_idx = sub_cluster1_idx_tensor.cpu().numpy()
                sub_cluster2_idx = sub_cluster2_idx_tensor.cpu().numpy()
                print(f"[Rank {self.local_rank}] e={e}, max_cluster={max_cluster}, "\
                f"sub1={len(sub_cluster1_idx)}, sub2={len(sub_cluster2_idx)}")

            # reassign samples in partition #2 to the empty class
            self.label_bank[sub_cluster2_idx] = e
            # update centroids of max_cluster and e
            self.update_centroids_memory([max_cluster, e])
            print("_redirect_empty_clusters", max_cluster, e)


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
            self.label_bank.numpy(), minlength=self.num_clusters)
        cluster_size = torch.tensor(histogram, device=self.centroids.device)
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
    def __init__(self, encoder, embedding_dim, num_sensors, num_clusters, train_dataloader, top_k=1):
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

    @torch.no_grad()
    def init_prototypes_with_data(self, device, num_clusters):
        """(수정된 버전) 텐서를 미리 할당하고, sample_ids를 이용해 채워 넣습니다."""
        self.eval()
        
        print("Pre-allocating tensor on GPU and collecting features...")

        # 1. (가장 중요) 데이터셋에서 '전체 샘플 수'를 안정적으로 가져옵니다.
        num_total_samples = len(self.train_dataloader().dataset)
        
        # 2. 이 크기에 맞춰 GPU에 빈 텐서를 '미리 할당'합니다.
        all_features_gpu = torch.empty(num_total_samples, self.clustering_manager.feature_dim, device=device)
        
        # 일부 데이터만 사용하여 특징 추출
        for i, (videos, sensors, labels, sample_ids) in enumerate(self.train_dataloader()):
                
            sensors = sensors.to(device)
            # sample_ids도 반드시 GPU 텐서여야 합니다.
            sample_ids = sample_ids.clone().to(dtype=torch.long, device=device)
            
            # 특징 추출 (GPU에서)
            _, features = self(sensors, return_features=True)
            representative_feature = self.get_representative_sensor_feature(sensors, labels, num_total_sensors=END_INDEX-START_INDEX+1, top_k=4, id=sample_ids)
            representative_feature = self.projection_layer(representative_feature)
            features += representative_feature

            # 3. 'sample_ids'를 인덱스로 사용하여, 계산된 features를 올바른 위치에 직접 삽입합니다.
            all_features_gpu[sample_ids, ...] = features

        self.clustering_manager.feature_bank = all_features_gpu.cpu()
        print("feature bank", self.clustering_manager.feature_bank.shape)

        # --- 이후 KMeans 로직은 동일 ---
        print("Moving features to CPU for KMeans...")
        all_features_cpu_numpy = all_features_gpu.cpu().numpy()
        
        print(f"Running KMeans with {len(all_features_cpu_numpy)} samples on CPU...")
        kmeans=self.clustering_manager.kmeans.fit(all_features_cpu_numpy)
        
        prototypes_gpu = torch.from_numpy(kmeans.cluster_centers_).to(device)
        self.clustering_manager.centroids.copy_(F.normalize(prototypes_gpu, dim=-1))

        # label bank 초기화
        print("Initializing label bank with KMeans results...")
        # 1. KMeans 결과를 PyTorch 텐서로 변환합니다. (레이블이므로 long 타입)
        initial_labels = torch.from_numpy(kmeans.labels_).long()
        print("initial_labels", initial_labels)
        # 2. 이 텐서를 label_bank에 복사하여 초기화합니다.
        #    .copy_()를 사용하여 텐서 내용을 바로 업데이트합니다.
        self.clustering_manager.label_bank = initial_labels
        
        print("Prototypes initialized.")
        self.clustering_manager.initialized = True
        self.train()
        return self.clustering_manager.centroids.clone()

    def forward(self, x, sample_ids=None, return_features=False, labels=None, step="train"):
        # x는 (B, C, T) 형태의 텐서
        
        features = self.encoder(x)["emb"]
        if step == "train" or step == "val":
            # print("x", x.shape)
            # representative_feature = torch.max(torch.abs(x), dim=-1).values
            # representative_feature = torch.quantile(torch.abs(x), q=0.99, dim=-1)
            # representative_feature = torch.var(x, dim=-1)
            # representative_feature = representative_feature * self.attention_weight
            # if labels is not None and self.global_rank == 0:  
            #     for i in range(len(labels)):
            #         print("representative_feature", representative_feature[i])
            #         print("ACTION_MERGE_LABELS", ACTION_MERGE_LABELS[labels[i].item()])
            #         if "Motion" in ACTION_MERGE_LABELS[labels[i].item()]:
            #             print("motion imu", x[i])
            # print("representative_feature", representative_feature.shape)
            # print("rule_feature", rule_feature.shape)
            # print("features", features.shape)
            # top_k_indices = torch.topk(representative_feature, k=self.top_k, dim=-1).indices
            # mask = torch.zeros_like(representative_feature)
            # mask.scatter_(1, top_k_indices, 1)
            # representative_feature = representative_feature * mask
        # 클러스터 유사도 점수 계산
            representative_feature = self.get_representative_sensor_feature(x, labels, num_total_sensors=END_INDEX-START_INDEX+1, top_k=self.top_k, id=sample_ids)
            representative_feature = self.projection_layer(representative_feature)
            features += representative_feature
            
        similarity_scores = self.clustering_manager.compute_similarity_scores(features)
        # similarity_scores = self.cls_head(features)

        # print("features", features)
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
    
        min_range, _ = torch.min(ranges, dim=1, keepdim=True)
        max_range, _ = torch.max(ranges, dim=1, keepdim=True)
        weighted_features = (ranges - min_range) / (max_range - min_range + 1e-8)
        # 정규화 x
        # weighted_features = ranges
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

    # --- 1. 센서 데이터 증강 (Data Augmentation) ---
    def time_warp(self, x, sigma=0.2, num_knots=4):
        
        # """시계열 데이터에 Time Warping 증강 적용 (간소화된 버전)"""
        B, C, T = x.shape
        device = x.device
        
        # 각 배치 아이템마다 다른 warping 적용
        warped_x = torch.zeros_like(x)
        for i in range(B):
            # 각 시간 포인트에 대한 랜덤 오프셋 생성
            time_indices = torch.arange(T, device=device, dtype=torch.float32)
            # 시간 왜곡 함수: sin 파형으로 자연스럽게 왜곡
            perturb = sigma * T * torch.sin(torch.linspace(0, 4*3.14, T, device=device))
            perturb = perturb * torch.rand(1, device=device)  # 배치마다 다른 강도
            
            # 왜곡된 인덱스 생성
            warped_indices = time_indices + perturb
            warped_indices = torch.clamp(warped_indices, 0, T-1)
            
            # 선형 보간으로 왜곡 적용
            warped_indices_low = warped_indices.floor().long()
            warped_indices_high = warped_indices.ceil().long()
            warped_indices_high = torch.clamp(warped_indices_high, 0, T-1)
            
            # 보간 가중치
            weight_high = warped_indices - warped_indices_low.float()
            weight_low = 1.0 - weight_high
            
            # 선형 보간으로 왜곡된 시계열 생성
            x_batch = x[i]  # (C, T)
            warped_x_batch = torch.zeros_like(x_batch)
            
            for t in range(T):
                low_idx = warped_indices_low[t]
                high_idx = warped_indices_high[t]
                warped_x_batch[:, t] = weight_low[t] * x_batch[:, low_idx] + weight_high[t] * x_batch[:, high_idx]
            
            warped_x[i] = warped_x_batch
        
        return warped_x
    
    def augment_imu_data(self, imu_data):
        return self.time_warp(imu_data)
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        
    @torch.no_grad()
    def evaluate(self, device):
        assert self.local_rank != 0 
        """ODC 모델을 DDP 환경에서 효율적으로 평가합니다."""
        print("evaluate_odc: Starting evaluation...")
        self.eval()

        # --- 1. 데이터 처리 (모든 Rank에서 실행) ---
        # 각 Rank에서 처리한 결과를 저장할 로컬 리스트
        features_list = []
        labels_list = []
        cluster_ids_list = []

        # DDP 환경에서는 DistributedSampler가 데이터를 분배합니다.
        for _, imu_data, labels, _ in self.train_dataloader():
            imu_data = imu_data.to(device)
            labels = labels.to(device)

            # 모델 순전파
            scores, features = self(imu_data, return_features=True, labels=labels, step="val")

            # 클러스터 할당
            cluster_ids = torch.argmax(scores, dim=1)

            features_list.append(features.detach())
            labels_list.append(labels.detach())
            cluster_ids_list.append(cluster_ids.detach())

        # --- 2. 로컬 결과 취합 (모든 Rank에서 실행) ---
        # 리스트에 담긴 텐서들을 하나의 텐서로 결합
        if len(features_list) > 0:
            features_local = torch.cat(features_list, dim=0)
            labels_local = torch.cat(labels_list, dim=0)
            cluster_ids_local = torch.cat(cluster_ids_list, dim=0)
        else:
            # 이 Rank에 할당된 데이터가 없는 경우 (데이터셋이 매우 작을 때 발생 가능)
            # 예외 처리가 필요하지만, 여기서는 데이터가 있다고 가정합니다.
            return None # 또는 적절한 빈 값 반환
        # if self.rank == 0:

        # --- 3. All Gather (모든 Rank의 결과를 Rank 0으로 모으기) ---
        # features_gathered = self.clustering_manager._gather(features_local)
        # labels_gathered = self.clustering_manager._gather(labels_local)
        # cluster_ids_gathered = self.clustering_manager._gather(cluster_ids_local)

    # --- 4. 최종 계산 및 로깅 (Rank 0에서만 실행) ---
        print("evaluate_odc: Rank 0 calculating results...")

        # # 모든 Rank의 결과를 하나의 Numpy 배열로 결합
        all_features = features_local.cpu().numpy()
        all_labels = labels_local.cpu().numpy()
        all_cluster_ids = cluster_ids_local.cpu().numpy()

        # 정확도 계산 로직
        print("Validation Accuracy Calculation")
        # compute_hungarian_matching 함수가 정의되어 있어야 함
        raw_accuracy, new_mapping = compute_hungarian_matching(all_cluster_ids, all_labels, self.clustering_manager.num_clusters)
        self.mapping = new_mapping # 매핑 업데이트

        mapped_cluster_ids = np.array([self.mapping.get(c, c) for c in all_cluster_ids])
        mapped_accuracy = np.mean(mapped_cluster_ids == all_labels)
        print(f"Val Accuracy with Existing Mapping: {mapped_accuracy:.4f}")

        wandb.log({
            "epoch": self.epoch,
            "val_accuracy_raw": raw_accuracy,
            "val_accuracy_mapped": mapped_accuracy
        })

        # 시각화 로직 (t-SNE)
        if True: # 시각화 플래그
            prototypes = self.clustering_manager.centroids.detach().cpu().numpy()

            # t-SNE 시각화 (DDP에서는 gather된 전체 데이터를 사용)
            # visualize_tsne 함수가 정의되어 있어야 함
            try:
                fig_2d, fig_3d = visualize_tsne(
                    all_features, all_labels, mapped_cluster_ids,
                    prototypes=prototypes,
                    title=f"ODC Validation at Epoch {self.epoch+1}",
                    num_classes=self.clustering_manager.num_clusters
                )

                wandb.log({
                    "val_tsne_2d": wandb.Image(fig_2d),
                    "val_tsne_3d": wandb.Image(fig_3d)
                })
                print("Validation TSNE 2D and 3D saved to wandb.")
                plt.close(fig_2d)
                plt.close(fig_3d)
            except Exception as e:
                print(f"Error during t-SNE visualization: {e}")

        # 모든 Rank가 평가를 마칠 때까지 대기 (다음 epoch으로 넘어가기 전 동기화)
        print("evaluate_odc: All ranks have finished evaluation.")

        # Rank 0에서만 결과 반환 (필요시)   
        return mapped_accuracy # 혹은 필요한 지표