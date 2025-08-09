import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection
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
    

class MW2StackRNNPooling(nn.Module):
    def __init__(self, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.GroupNorm(2, 6),
            Block(6, input_dim, 10),
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
    
    def __init__(self, num_classes: int):
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

        # ==========================================================================================
        # 1단계: 특징 추출 (재료 준비)
        # ==========================================================================================
        visual_output = self.video_model(video_reshaped, output_hidden_states=True)
        
        # 중간 특징과 최종 특징 추출
        hidden_states = visual_output.hidden_states
        intermediate_features = hidden_states[6]
        final_features = hidden_states[-1]

        seq_len = intermediate_features.shape[1]
        hidden_size = intermediate_features.shape[2]

        # 특징 맵의 형태를 (B, T, Seq_Len, Hidden_Size)로 복원        
        # 각 프레임별 특징 맵 스택을 반환
        intermediate_features = intermediate_features.view(batch_size, n_frames, seq_len, hidden_size)
        final_features = final_features.view(batch_size, n_frames, seq_len, hidden_size)

        return {
            "intermediate_features": intermediate_features,
            "final_features": final_features
        }


#################################################################


class CAMGenerator(nn.Module):
    """
    ViT 특징 맵으로부터 1x1 Conv를 사용하여 프레임별 CAM과 Logits를 생성합니다.
    """
    def __init__(self, input_hidden_size: int, num_classes: int):
        super().__init__()
        # 1x1 Convolution 레이어를 분류기로 사용합니다.
        # in_channels: ViT의 hidden_size
        # out_channels: 분류할 클래스의 수
        self.classifier = nn.Conv2d(
            in_channels=input_hidden_size,
            out_channels=num_classes,
            kernel_size=1
        )

        self.pooling = RankedTopKPooling(k1_ratio=0.05)

    def forward(self, features: torch.Tensor):
        # 입력 features shape: (B, T, Seq_Len, Hidden_Size)
        # e.g., (8, 16, 50, 768) -> 50 = 49(패치) + 1(CLS)

        batch_size, n_frames, seq_len, hidden_size = features.shape

        # 배치와 프레임 차원을 합쳐서 처리 효율을 높입니다.
        # (B, T, Seq_Len, Hidden_Size) -> (B * T, Seq_Len, Hidden_Size)
        features = features.view(batch_size * n_frames, seq_len, hidden_size)

        # --- 1. 패치 토큰 분리 ---
        # CAM은 공간 정보를 담고 있으므로, CLS 토큰(인덱스 0)을 제외한 패치 토큰만 사용합니다.
        # (B*T, Seq_Len, Hidden_Size) -> (B*T, Num_Patches, Hidden_Size)
        patch_features = features[:, 1:, :]
        num_patches = patch_features.shape[1] # e.g., 49

        # --- 2. Conv2d 입력을 위한 형태 변환 ---
        # 패치 그리드의 크기를 계산합니다 (정사각형이라고 가정).
        patch_grid_size = int(num_patches ** 0.5) # e.g., 7

        # (B*T, Num_Patches, Hidden_Size) -> (B*T, Hidden_Size, Num_Patches)
        patch_features = patch_features.permute(0, 2, 1)

        # (B*T, Hidden_Size, Num_Patches) -> (B*T, Hidden_Size, H_patch, W_patch)
        patch_features_2d = patch_features.view(
            batch_size * n_frames, hidden_size, patch_grid_size, patch_grid_size
        )

        # --- 3. 1x1 Conv를 이용한 CAM 생성 ---
        # (B*T, Hidden_Size, H, W) -> (B*T, Num_Classes, H, W) -> [256, 10, 7, 7]
        cam = self.classifier(patch_features_2d)

        # --- 4. 프레임별 Logits 계산 ---
        # logits의 shape은 (256, 10)
        logits = self.pooling(cam)

        # --- (수정) 시각화를 위한 ReLU 적용 ---
        # Logits 계산이 끝난 후, 시각화 품질을 높이기 위해 CAM에 ReLU를 적용합니다.
        # 이 단계는 예측 결과(logits)에 영향을 주지 않습니다.
        cam_for_visualization = F.relu(cam)

        # --- 5. 최종 출력 형태 복원 ---
        # (B*T, Num_Classes) -> (B, T, Num_Classes)
        logits = logits.view(batch_size, n_frames, -1)

        # (B*T, Num_Classes, H, W) -> (B, T, Num_Classes, H, W)
        # 최종적으로 반환되는 CAM은 ReLU가 적용된 시각화용 CAM입니다.
        cam_for_visualization = cam_for_visualization.view(
            batch_size, n_frames, -1, patch_grid_size, patch_grid_size
        )

        # 반환 값의 'cam' 키에 ReLU가 적용된 CAM을 할당합니다.
        return {"logits": logits, "cam": cam_for_visualization}


#################################################################


class ViTWithCAM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # 1. 특징 추출기: LoRA가 적용된 Clip4ClipVisionModel을 그대로 사용합니다.
        # 이 모델 내부의 classifier는 무시하고, 특징 추출 결과만 사용합니다.
        self.feature_extractor = Clip4ClipVisionModel(num_classes=num_classes)

        # 2. CAM 생성기: ViT가 추출한 특징을 받아 CAM과 최종 logits를 생성합니다.
        hidden_size = self.feature_extractor.video_model.config.hidden_size # 768
        
        self.cam_generator = CAMGenerator(input_hidden_size=hidden_size, num_classes=num_classes)

    def forward(self, video: torch.Tensor):
        
        # 1. LoRA가 적용된 ViT를 통해 특징을 추출합니다.
        # 이 과정에서 그래디언트가 LoRA 파라미터로 흘러가도록 합니다.
        # features_dict --> {"intermediate_features": intermediate_features, "final_features": final_features}
        features_dict = self.feature_extractor(video)
        
        # ViT의 레이어에서 나온 특징맵을 사용합니다.
        # Shape: (B, T, Seq_Len, Hidden_Size)
        final_features = features_dict["final_features"]
        intermediate_features = features_dict["intermediate_features"]
        
        # 2. 추출된 특징맵을 CAM 생성기에 전달하여 최종 출력(logits, cam)을 얻습니다.
        # 이 모듈의 파라미터는 전체가 학습됩니다 (full-tuning).
        # output_dict --> {"logits": logits, "cam": activated_cam}
        output_dict = self.cam_generator(final_features)
        
        output_dict["intermediate_features"] = intermediate_features
        output_dict["final_features"] = final_features
        
        return output_dict
        
        
#################################################################


class RankedTopKPooling(nn.Module):

    def __init__(self, k1_ratio):
        super().__init__()
        self.k1_ratio = k1_ratio

    def forward(self, features: torch.Tensor):
        
        # 입력 features shape: (256, 10, 7, 7)
        bt, c, h, w = features.shape

        # --- 1단계: 공간적 풀링 ---
        # 공간 차원을 하나로 합침: (256, 10, 7, 7) --> (256, 10, 49)
        # 각 프레임은 49개의 공간적 위치(패치)를 가짐
        spatial_features = features.view(bt, c, h * w)

        # k1 값 계산 
        k1 = max(1, int((h * w) * self.k1_ratio))

        # 공간 차원에서 top-k 값을 찾음
        # 49개의 패치 중에서 활성화 값이 가장 높은 k1개의 패치들만 선택
        # topk_spatial_values shape: (B, C, D, k1)
        topk_spatial_values, _ = torch.topk(spatial_features, k=k1, dim=-1)

        # k1개의 값들만 평균내어 공간 정보를 압축
        # spatial_pooled shape: [256, 10]
        spatial_pooled = torch.mean(topk_spatial_values, dim=-1)

        return spatial_pooled
    

#################################################################













