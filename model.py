import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, AutoModel
from peft import LoraConfig, get_peft_model
import random
from einops import rearrange, repeat


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
    

class SensorModel(nn.Module):
    def __init__(self, sensor_channels, input_dim=32, size_embeddings: int = 128):
        super().__init__()

        num_groups_for_input = 1

        self.backbone = nn.Sequential(
            nn.GroupNorm(num_groups_for_input, sensor_channels),
            Block(sensor_channels, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5, pool_type="adaptive", embedding_size=32),
            nn.GroupNorm(4, input_dim),
            nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )

    def forward(self, batch):
        # GRU는 (output, h_n) 형태의 튜플을 반환합니다.
        # h_n (마지막 타임스텝의 은닉 상태)을 사용합니다.
        # h_n의 shape: (num_layers, batch_size, hidden_size)
        _, last_hidden_state = self.backbone(batch)
        
        # GRU 레이어가 하나이므로 첫 번째 요소를 선택하고, batch 차원을 유지하기 위해 squeeze(0) 대신 [0]을 사용합니다.
        emb = last_hidden_state[0] # Shape: (batch_size, hidden_size)        
        
        out = {"emb": emb}
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
        """
        Args:
            image_size (int): 입력 이미지의 크기 (H 또는 W).
            num_classes (int): 최종 분류할 클래스의 수.
            target_size (tuple): Attention Bridge가 출력할 특징 맵의 크기.
        """
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
        encoder_mid_channels = 256
        appearance_out_channels = 128
        motion_out_channels = 128  # RNN의 입력 크기가 됩니다.
        motion_rnn_hidden_size = 128

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
        """
        전체 모델의 순전파 파이프라인을 실행합니다.
        Args:
            video (torch.Tensor): (B, T, C, H, W) 형태의 원본 비디오.
        Returns:
            dict: 모델의 출력을 담은 딕셔너리.
                    - "logits": 최종 분류 결과 (prediction).
                    - "v_appearance": 추출된 외형 벡터.
                    - "v_motion": 추출된 동작 벡터.
                    - "transformed_features": Attention Bridge의 출력 특징 맵.
        """
        features_dict = self.feature_extractor(video)

        # 1. 특징 추출 (F_A)
        final_features = features_dict["final_features"]

        # 2. Attention Bridge를 통해 변환된 특징 맵 (F'_A) 생성
        # transformed_features shape: (B, T, D, H_t, W_t)
        transformed_features = self.attention_bridge(final_features)

        transformed_features = transformed_features.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)

        # 3. Appearance 및 Motion 벡터 추출
        v_appearance = self.appearance_encoder(transformed_features) # [B, 128]
        v_motion = self.motion_encoder(transformed_features) # [B, 128]

        # --- 최종 출력 통합 ---
        final_output = {
            "v_appearance": v_appearance,
            "v_motion": v_motion
        }

        return final_output