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

from visualization import visualize_tsne
from utils import compute_hungarian_matching


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
        self.regressor.weight.data.zero_()
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

    def __init__(self, input_hidden_size: int, patch_grid_size: int, target_size: tuple = (112, 112)):
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
    """
    (수정 최종 버전) 특징 추출, Attention Bridge, 그리고 Appearance/Motion 인코딩을 모두 포함하는 통합 모델.
    """
    def __init__(self, image_size: int, target_size: tuple = (112, 112)):
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
        motion_out_channels = 256  # RNN의 입력 크기가 됩니다.
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
            "v_motion": v_motion
        }

        return final_output


#################################################################


# --- ODC 메모리 뱅크 관리자 ---
class ClusteringManager(nn.Module):
    def __init__(self, num_clusters, feature_dim, momentum=0.99, temperature=0.1, device='cuda'):
        super().__init__()

        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        
        # 클러스터 중심점(centroids) 초기화 - 더 넓게 분포되도록 초기화
        # 각 차원마다 균등 분포를 사용하여 더 잘 분산되도록 함
        centroids = torch.rand(num_clusters, feature_dim, device=device) * 2.0 - 1.0  # [-1, 1] 범위의 균등 분포
                
        # 직교성을 높이기 위한 추가 처리
        # QR 분해를 통해 직교 벡터 얻기
        if num_clusters <= feature_dim:  # 클러스터 수가 차원보다 작거나 같을 때만 가능
            q, r = torch.linalg.qr(centroids.t())  # 직교 행렬 Q 얻기
            centroids = q[:, :num_clusters].t()  # 직교 벡터로 중심점 설정
        self.register_buffer('centroids', centroids)
        
        self.cluster_size = torch.zeros(num_clusters, device=device)
        
        # Pseudo label 메모리 뱅크 (ODC 논문과 유사하게 중앙화된 방식으로 관리)
        # 각 샘플을 고유하게 식별할 수 있는 ID를 저장하기 위한 딕셔너리
        # key: sample_id, value: pseudo_label
        self.memory_bank = {}
        
        # 전체 데이터셋에 대한 특징 메모리 뱅크 (샘플 ID -> 특징 벡터)
        self.feature_bank = {}
        
        # 메모리 뱅크 초기화 여부를 추적
        self.memory_initialized = False
        
        # 클러스터 재분배 후 메모리 뱅크 업데이트 지연을 위한 플래그
        self.pending_memory_update = False
        
        # 빈 클러스터 감지와 재할당을 위한 임계값
        self.min_cluster_size = 10  # 이 값보다 작으면 비어있다고 간주
        
        # 가중치 계산을 위한 상수
        self.class_weight_power = 1.0  # 클러스터 크기에 적용할 거듭제곱
        
        
    @torch.no_grad()
    def update_centroids_with_momentum(self, features, cluster_ids):
        """모멘텀 방식으로 중심점만 업데이트합니다.
        클러스터 크기는 update_memory_bank에서 전체 데이터셋을 기준으로 계산됩니다."""

        for k in range(self.num_clusters):
            # 현재 클러스터에 할당된 특징들 선택
            mask = (cluster_ids == k)
            cluster_samples = mask.sum()
            
            if cluster_samples > 0:
                # 현재 클러스터에 할당된 특징들의 평균
                new_centroid = features[mask].mean(dim=0)
                # new_centroid, _ = torch.median(features[mask], dim=0)
                # 중심점만 모멘텀으로 업데이트
                self.centroids[k] = self.momentum * self.centroids[k] + (1 - self.momentum) * new_centroid
                # 정규화는 한 번만 수행 (중요: 정규화는 특징 차원을 따라 dim=0이 아니라 dim=-1 사용)
                # self.centroids[k] = F.normalize(self.centroids[k], dim=-1)
                
        # 참고: 클러스터 크기(self.cluster_size)는 update_memory_bank에서 전체 데이터셋을 기준으로 계산됨


    def compute_similarity_scores(self, features):
        """특징과 중심점 간의 유사도 점수를 계산합니다."""

        # 코사인 유사도 계산 (L2 정규화 후 내적)
        features_norm = F.normalize(features, dim=1)
        centroids_norm = F.normalize(self.centroids, dim=1)
        similarity = torch.mm(features_norm, centroids_norm.t())
        
        # 온도 파라미터 적용
        return similarity / self.temperature


    def compute_class_weights(self):
        """클러스터 크기에 근거한 클래스 가중치를 계산합니다."""

        # 클러스터 크기가 0인 경우를 방지하기 위한 정규화
        normalized_sizes = self.cluster_size + 1e-8
        
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

        
    def get_largest_cluster(self):
        """가장 큰 클러스터와 그 크기를 반환합니다.
        최소 샘플 개수를 보장하여 실질적으로 분할 가능한 클러스터를 반환합니다."""
        # 최소 샘플 수 기준 (최소 10개 이상은 있어야 분할 의미가 있음)
        min_samples_for_split = 10
        valid_clusters = torch.where(self.cluster_size >= min_samples_for_split)[0]
        
        if len(valid_clusters) == 0:
            # 모든 클러스터가 작으면 그나마 가장 큰 것 반환
            largest_idx = torch.argmax(self.cluster_size)
        else:
            # 충분히 큰 클러스터 중에서 가장 큰 것 선택
            largest_idx = valid_clusters[torch.argmax(self.cluster_size[valid_clusters])]
            
        return largest_idx, self.cluster_size[largest_idx]

        
    def get_empty_clusters(self):
        """빈 클러스터(임계값 미만)를 반환합니다."""
        # 클러스터 크기가 임계값보다 작거나, 상대적으로 너무 작은 클러스터 찾기
        avg_size = self.cluster_size.mean().item()
        relative_threshold = avg_size * 0.2  # 평균의 20% 미만인 클러스터도 빈 것으로 간주
        threshold = min(self.min_cluster_size, relative_threshold)
        return torch.where(self.cluster_size < threshold)[0]

        
    def redistribute_cluster(self, empty_idx, largest_idx, features, labels):
        """크기가 큰 클러스터를 분할하여 빈 클러스터를 재활용합니다."""
        # 가장 큰 클러스터에 속한 샘플들을 찾기
        largest_cluster_samples = (labels == largest_idx)
        largest_cluster_features = features[largest_cluster_samples]
        
        # 실제 샘플 수 확인
        sample_count = len(largest_cluster_features)
        
        # 최소 4개 이상의 샘플이 있어야 의미 있는 분할이 가능함
        if sample_count < 4:
            print(f"Warning: Largest cluster {largest_idx} has only {sample_count} samples. Cannot redistribute.")
            return False
            
        # 클러스터 내에서 2개의 서브클러스터로 분할
        # 최대한 다양한 분할을 위해 n_init 값을 높이고, random_state를 다르게 설정
        try:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=np.random.randint(0, 1000)).fit(largest_cluster_features.cpu().numpy())
            sub_labels = torch.tensor(kmeans.labels_, device=self.device)
            
            # 재분배 결과 확인 (최소한 1개 이상의 샘플이 각 서브클러스터에 할당되었는지)
            sub_counts = [(sub_labels == i).sum().item() for i in range(2)]
            if min(sub_counts) < 1:
                print(f"Warning: Subcluster division resulted in imbalanced clusters: {sub_counts}. Cannot redistribute.")
                return False
        except Exception as e:
            print(f"Error during KMeans clustering: {e}. Cannot redistribute.")
            return False
        
        # 서브클러스터 중심점 계산
        sub_centroids = []
        for i in range(2):
            sub_centroid = largest_cluster_features[sub_labels == i].mean(dim=0)
            # 정규화 시 dim=-1 사용
            sub_centroids.append(F.normalize(sub_centroid, dim=-1))
        
        # 하나는 기존 클러스터로, 하나는 빈 클러스터로 할당
        self.centroids[largest_idx] = sub_centroids[0]
        self.centroids[empty_idx] = sub_centroids[1]
        
        # 클러스터 크기 업데이트 (위에서 이미 계산한 sub_counts 사용)
        self.cluster_size[largest_idx] = sub_counts[0]
        self.cluster_size[empty_idx] = sub_counts[1]
        
        print(f"Redistributed cluster: Split cluster {largest_idx} ({sub_counts[0]} samples) "
            f"and reassigned {sub_counts[1]} samples to empty cluster {empty_idx}")
            
        # 만약 재분배 후에도 여전히 작은 클러스터가 있다면 로그로 알리기
        if min(sub_counts) < self.min_cluster_size:
            print(f"Warning: After redistribution, one of the clusters still has fewer than {self.min_cluster_size} samples ({min(sub_counts)}).")
            
        # 메모리 뱅크 업데이트 필요 (외부에서 처리)
        
        return True


#################################################################


# --- ODC 모델 ---
class ClusteringModel(nn.Module):
    def __init__(self, encoder, embedding_dim, num_sensors, num_clusters):
        super().__init__()

        self.encoder = encoder
    
        # ODC 관리자
        self.clustering_manager = ClusteringManager(num_clusters=num_clusters, feature_dim=embedding_dim)
        self.projection_layer = nn.Linear(num_sensors, embedding_dim)
        self.epoch = 0
        
    def forward(self, x, sample_ids=None, return_features=False, labels=None, step="train"):        
        features = self.encoder(x)["emb"]
        
        if step == "train":
            rule_feature = self.get_representative_sensor_feature(x, labels, num_total_sensors=97, top_k=4, id=sample_ids)
            rule_feature = self.projection_layer(rule_feature)    
            features += rule_feature * 1.0/(self.epoch+1)

        else:
            self.epoch += 1

        similarity_scores = self.clustering_manager.compute_similarity_scores(features)
        
        if return_features:
            return similarity_scores, features

        return similarity_scores


    # 전체 데이터셋의 pseudo label을 계산하고 메모리 뱅크에 저장하는 함수
    @torch.no_grad()
    def init_memory_bank(self, dataloader, device):
        """전체 데이터셋에 대한 pseudo label과 특징을 계산하고 메모리 뱅크에 저장합니다.
        ODC 논문과 유사하게 전체 데이터셋에 대한 중앙화된 메모리 뱅크를 관리합니다."""
        self.eval()
        
        # 전체 샘플 수와 업데이트된 샘플 수 추적
        total_samples = 0
        updated_samples = 0
        
        # 클러스터 할당 카운터 (전체 데이터셋에 대한 클러스터 크기 계산용)
        cluster_counts = torch.zeros(self.clustering_manager.num_clusters, device=device)
        
        for batch in dataloader:
            imu_data = batch['imu'].to(device)
            sample_ids = batch['video_id'] if 'video_id' in batch else [f"sample_{i+total_samples}" for i in range(len(imu_data))]
            total_samples += len(imu_data)
            
            # 모델 순전파 (특징 벡터도 얻기)
            scores, features = self(imu_data, return_features=True)
            
            # 클러스터 할당 (argmax)
            current_cluster_ids = torch.argmax(scores, dim=1)
            
            # 클러스터 할당 카운트 (히스토그램)
            for cluster_id in range(self.clustering_manager.num_clusters):
                cluster_counts[cluster_id] += (current_cluster_ids == cluster_id).sum()
            
            # 메모리 뱅크에 pseudo label과 특징 벡터 저장
            self.update_memory_bank(sample_ids, current_cluster_ids, features)
            updated_samples += len(sample_ids)
            
        # 전체 데이터셋에 대한 클러스터 크기 업데이트 (ODC 논문 방식)
        self.clustering_manager.cluster_size = cluster_counts
        
        # 클러스터 크기 정보 출력
        print(f"Cluster size distribution after memory bank update:")
        for i, count in enumerate(cluster_counts.cpu().numpy()):
            print(f"  Cluster {i}: {count:.0f} samples")
        
        # 메모리 뱅크 초기화 완료 표시
        self.clustering_manager.memory_initialized = True
        print(f"Memory bank initialized with {updated_samples} samples.")
        
        self.train()

    
    @torch.no_grad()
    def update_centroids(self, features, cluster_ids):
        """ODC Manager의 중심점을 업데이트합니다."""
        self.clustering_manager.update_centroids_with_momentum(features, cluster_ids)

    
    @torch.no_grad()
    def update_memory_bank(self, sample_ids, pseudo_labels, features=None):
        """샘플별 pseudo label과 특징을 메모리 뱅크에 저장합니다.
        ODC 논문에서처럼 전체 데이터셋에 대한 메모리를 유지합니다."""
        for idx, sample_id in enumerate(sample_ids):
            self.clustering_manager.memory_bank[sample_id] = pseudo_labels[idx].item()
            
            # 특징 벡터도 저장 (제공된 경우)
            if features is not None:
                self.clustering_manager.feature_bank[sample_id] = features[idx].detach()


    def get_pseudo_labels(self, sample_ids):
        """메모리 뱅크에서 샘플에 대한 pseudo label을 조회합니다."""
        device = next(self.parameters()).device
        pseudo_labels = []
        
        # 각 샘플 ID에 대해 저장된 pseudo label 조회
        for sample_id in sample_ids:
            if sample_id in self.clustering_manager.memory_bank:
                pseudo_labels.append(self.clustering_manager.memory_bank[sample_id])
            else:
                # 메모리 뱅크에 없는 경우 -1 (무시할 값) 반환
                pseudo_labels.append(-1)
        
        return torch.tensor(pseudo_labels, device=device)


    # Top-K 센서 처리    
    def get_representative_sensor_feature(self, imu_batch, labels, num_total_sensors=97, top_k=1, id=None):
        """
        각 샘플에서 신호 변화가 가장 큰 센서를 찾아 원-핫 벡터로 만듭니다.
        imu_batch: (B, C, L) 형태의 텐서
        """
        
        # 각 채널(센서)의 분산 계산 (max - min)
        ranges = torch.var(imu_batch, dim=2)

        # 가장 분산이 큰 센서의 인덱스 찾기 (분산이 0인것 제외)    
        min_range, _ = torch.min(ranges, dim=1, keepdim=True)
        max_range, _ = torch.max(ranges, dim=1, keepdim=True)
        weighted_features = (ranges - min_range) / (max_range - min_range + 1e-8)

        # Top-K에 해당하지 않는 값들을 0으로 마스킹
        # 가장 큰 Top-K 값만 남기고 나머지는 0으로 만들기 위한 마스크 생성
        _, top_indices = torch.topk(weighted_features, k=top_k, dim=1)
        mask = torch.zeros_like(weighted_features)
        mask.scatter_(1, top_indices, 1)

        # 마스크를 적용하여 최종 특징 생성
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


    def evaluate_odc(self, dataloader, device, cluster_mapping=None, epoch=0, use_wandb=True):
        """ODC 모델을 평가합니다."""
        self.eval()
        all_features = []
        all_labels = []
        all_cluster_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                imu_data = batch['imu'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else None
                
                # 모델 순전파
                scores, features = self(imu_data, return_features=True, labels=labels, step="val")
                
                # 클러스터 할당
                cluster_ids = torch.argmax(scores, dim=1)
                
                if labels is not None:
                    all_features.append(features.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())
                    all_cluster_ids.append(cluster_ids.detach().cpu().numpy())
        
        # 정확도 계산
        if all_labels:
            all_features = np.vstack(all_features)
            all_labels = np.concatenate(all_labels)
            all_cluster_ids = np.concatenate(all_cluster_ids)
            
            # 매핑 없이 정확도 계산
            raw_accuracy, new_mapping = compute_hungarian_matching(all_cluster_ids, all_labels, self.clustering_manager.num_clusters)
            
            # 기존 매핑 적용 시 정확도
            if cluster_mapping is not None:
                mapped_cluster_ids = np.array([cluster_mapping.get(c, c) for c in all_cluster_ids])
                mapped_accuracy = np.mean(mapped_cluster_ids == all_labels)
                print(f"Val Accuracy with Existing Mapping: {mapped_accuracy:.4f}")
                
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "val_accuracy_raw": raw_accuracy,
                        "val_accuracy_mapped": mapped_accuracy
                    })
            else:
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "val_accuracy": raw_accuracy
                    })
            
            # 시각화 (선택적)
            if epoch % 2 == 0:
                # 프로토타입 가져오기
                prototypes = self.odc_manager.centroids.detach().cpu().numpy()
                
                # 메모리 뱅크에서 pseudo label 가져오기
                memory_bank_labels = []
                sample_ids = []
                
                # 다시 데이터를 반복하면서 샘플 ID 수집
                for batch in dataloader:
                    if 'video_id' in batch:
                        sample_ids.extend(batch['video_id'])
                    else:
                        # 샘플 ID가 없는 경우 임의로 생성
                        sample_ids.extend([f"val_sample_{i+len(sample_ids)}" for i in range(len(batch['imu']))])
                
                # ODC 논문과 유사하게 메모리 뱅크에서 직접 특징과 라벨을 추출
                if self.odc_manager.memory_initialized:
                    # 메모리 뱅크에서 특징과 라벨 추출
                    memory_features = []
                    memory_labels = []
                    
                    # 메모리 뱅크에서 샘플 ID 가져오기
                    all_sample_ids = list(self.odc_manager.memory_bank.keys())
                    
                    # 검증 데이터만 필터링 (샘플 ID 접두사로 구분)
                    val_sample_ids = [sid for sid in all_sample_ids if sid.startswith("val_")]
                    
                    if len(val_sample_ids) > 0:
                        # 메모리 뱅크에서 라벨과 특징 추출
                        for sid in val_sample_ids:
                            memory_labels.append(self.odc_manager.memory_bank[sid])
                            if sid in self.odc_manager.feature_bank:
                                memory_features.append(self.odc_manager.feature_bank[sid].cpu().numpy())
                        
                        if len(memory_features) > 0:
                            # 특징과 라벨을 numpy 배열로 변환
                            memory_features = np.array(memory_features)
                            memory_labels = np.array(memory_labels)
                            
                            # 매핑 적용
                            if cluster_mapping is not None:
                                mapped_memory_labels = np.array([cluster_mapping.get(c, c) for c in memory_labels])
                            else:
                                mapped_memory_labels = np.array([new_mapping.get(c, c) for c in memory_labels])
                            
                            # 실제 라벨 (있는 경우)
                            if len(all_labels) > 0:
                                # 메모리 뱅크는 전체 데이터셋을 포함하므로 all_labels와 길이가 다를 수 있음
                                vis_labels = np.zeros(len(memory_features), dtype=np.int64)
                                vis_labels[:min(len(all_labels), len(vis_labels))] = all_labels[:min(len(all_labels), len(vis_labels))]
                            else:
                                vis_labels = np.zeros(len(memory_features), dtype=np.int64)
                            
                            # t-SNE 시각화 (2D와 3D 모두)
                            fig_2d, fig_3d = visualize_tsne(
                                memory_features, vis_labels, mapped_memory_labels,
                                prototypes=prototypes,
                                title=f"ODC Validation at Epoch {epoch+1} (Memory Bank)",
                                num_classes=self.odc_manager.num_clusters
                            )
                        else:
                            # 메모리 뱅크에 특징이 없는 경우 기존 방식 사용
                            if cluster_mapping is not None:
                                mapped_ids = np.array([cluster_mapping.get(c, c) for c in all_cluster_ids])
                            else:
                                mapped_ids = np.array([new_mapping.get(c, c) for c in all_cluster_ids])
                            
                            fig_2d, fig_3d = visualize_tsne(
                                all_features, all_labels, mapped_ids,
                                prototypes=prototypes,
                                title=f"ODC Validation at Epoch {epoch+1} (No Memory Features)",
                                num_classes=self.odc_manager.num_clusters
                            )
                    else:
                        # 검증 샘플이 메모리 뱅크에 없는 경우 기존 방식 사용
                        if cluster_mapping is not None:
                            mapped_ids = np.array([cluster_mapping.get(c, c) for c in all_cluster_ids])
                        else:
                            mapped_ids = np.array([new_mapping.get(c, c) for c in all_cluster_ids])
                        
                        fig_2d, fig_3d = visualize_tsne(
                            all_features, all_labels, mapped_ids,
                            prototypes=prototypes,
                            title=f"ODC Validation at Epoch {epoch+1} (No Val Samples in Memory)",
                            num_classes=self.odc_manager.num_clusters
                        )
                else:
                    # 메모리 뱅크가 초기화되지 않은 경우 기존 방식 사용
                    if cluster_mapping is not None:
                        mapped_ids = np.array([cluster_mapping.get(c, c) for c in all_cluster_ids])
                    else:
                        mapped_ids = np.array([new_mapping.get(c, c) for c in all_cluster_ids])
                    
                    fig_2d, fig_3d = visualize_tsne(
                        all_features, all_labels, mapped_ids,
                        prototypes=prototypes,
                        title=f"ODC Validation at Epoch {epoch+1}",
                        num_classes=self.odc_manager.num_clusters
                    )
                
                if use_wandb:
                    # 일관된 wandb 키 사용
                    wandb.log({
                        "val_tsne_2d": wandb.Image(fig_2d),
                        "val_tsne_3d": wandb.Image(fig_3d)
                    })
                plt.close(fig_2d)
                plt.close(fig_3d)
        
        return all_features, all_labels, all_cluster_ids