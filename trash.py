# class CAMGenerator(nn.Module):
#     def __init__(self, input_hidden_size: int, num_classes: int, base_f: int = 128):
#         super().__init__()

#         # MLP 블록들은 그대로 사용
#         self.mlp_blocks = nn.ModuleList([
#             MLP3D(input_hidden_size, base_f),
#             MLP3D(base_f, base_f * 2),
#             MLP3D(base_f * 2, base_f * 4)
#         ])

#         concatenated_channels = input_hidden_size + base_f + (base_f * 2) + (base_f * 4)

#         # 2. 논문의 연산 순서에 맞게 풀링 모듈과 분류기 재정의
#         # (수정) 풀링을 먼저 하므로, 분류기는 Conv3d가 아닌 Linear가 됨
#         self.ranked_pooling = RankedTopKPooling(k1_ratio=0.1, k2_ratio=0.4)
#         self.classifier = nn.Linear(concatenated_channels, num_classes)

#         # 시각화를 위한 CAM 활성화 함수
#         self.cam_activation = nn.ReLU(inplace=True)


#     def forward(self, features: torch.Tensor):
#         batch_size, n_frames, seq_len, hidden_size = features.shape
#         patch_features = features[:, :, 1:, :]
#         h_w = int((seq_len - 1) ** 0.5)
#         patch_features_reshaped = patch_features.view(batch_size, n_frames, h_w, h_w, hidden_size)
#         features_for_processing = patch_features_reshaped.permute(0, 4, 1, 2, 3)

#         intermediate_features_out = []
#         out = features_for_processing
#         for mlp_block in self.mlp_blocks:
#             out = mlp_block(out)
#             intermediate_features_out.append(out)

#         last_feature_map = intermediate_features_out[-1]
#         target_size = last_feature_map.shape[2:]
#         feature_maps_to_concat = []
#         resized_initial_features = F.interpolate(features_for_processing, size=target_size, mode='trilinear', align_corners=False)
#         feature_maps_to_concat.append(resized_initial_features)

#         for feature_map in intermediate_features_out:
#             resized_map = F.interpolate(feature_map, size=target_size, mode='trilinear', align_corners=False)
#             feature_maps_to_concat.append(resized_map)

#         # cat_feature shape: (B, C_concat, D, H, W)
#         cat_feature = torch.cat(feature_maps_to_concat, dim=1)


#         # --- Logits 및 CAM 계산 흐름 수정 ---

#         # 1. (수정) 특성 맵에 '순위 기반 풀링'을 먼저 적용
#         # pooled_features shape: (B, C_concat)
#         pooled_features = self.ranked_pooling(cat_feature)        

#         # 2. (수정) 압축된 벡터를 Linear Classifier에 통과시켜 Logits 생성
#         logits = self.classifier(pooled_features)

#         # 3. (수정) CAM 생성: 풀링 전 특성 맵과 분류기 가중치를 사용
#         # self.classifier.weight shape: (num_classes, C_concat)
#         # 가중치를 사용하여 특성 맵의 채널들에 대한 가중 합을 계산
#         # cam shape: (B, num_classes, D, H, W)
#         cam = F.conv3d(
#             cat_feature,
#             self.classifier.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         )
        
#         # 시각화를 위해 CAM에 ReLU 적용
#         activated_cam = self.cam_activation(cam)

#         return {
#             "logits": logits,
#             "cam": activated_cam
#         }








# class MLP3D(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, 
#                 mlp_ratio: int = 4, dropout: float = 0.0):

#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         hidden_channels = int(in_channels * mlp_ratio)

#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_channels, out_channels),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         # Get original shape
#         B, C, D, H, W = x.shape

#         # Reshape for MLP: (B, C, D, H, W) -> (B, D*H*W, C)
#         x_reshaped = x.flatten(2).permute(0, 2, 1)

#         # Apply MLP
#         x_mlp = self.mlp(x_reshaped)

#         # Reshape back to 5D tensor: (B, D*H*W, C_out) -> (B, 
#         # C_out, D*H*W) -> (B, C_out, D, H, W)
#         x_out = x_mlp.permute(0, 2, 1).view(B, self.out_channels, D, H, W)

#         return x_out




# class CAMGenerator(nn.Module):
#     def __init__(self, input_hidden_size: int, num_classes: int, base_f: int = 128):
#         super().__init__()

#         # 1. Conv3D 블록
#         self.conv_blocks = nn.ModuleList([
#             Conv3DBlock(input_hidden_size, base_f),
#             Conv3DBlock(base_f, base_f * 2),
#             Conv3DBlock(base_f * 2, base_f * 4)
#         ])

#         # 2. 다중 레벨 특징 융합을 위한 채널 수 계산
#         concatenated_channels = input_hidden_size + base_f + (base_f * 2) + (base_f * 4)

#         # 3. 순위 기반 풀링 모듈
#         self.ranked_pooling = RankedTopKPooling(k1_ratio=0.05, k2_ratio=1)

#         # 4. 최종 분류기
#         self.classifier = nn.Linear(concatenated_channels, num_classes)

#         # 5. 시각화를 위한 CAM 활성화 함수
#         self.cam_activation = nn.ReLU(inplace=True)


#     def forward(self, features: torch.Tensor):

#         # --- 1. 입력 준비 ---
#         batch_size, n_frames, seq_len, hidden_size = features.shape
#         patch_features = features[:, :, 1:, :]
#         h_w = int((seq_len - 1) ** 0.5)
#         patch_features_reshaped = patch_features.view(batch_size, n_frames, h_w, h_w, hidden_size)
#         features_for_processing = patch_features_reshaped.permute(0, 4, 1, 2, 3)

#         # --- 2. 계층적 특징 추출 ---
#         intermediate_features_out = []
#         out = features_for_processing
        
#         for conv_block in self.conv_blocks:
#             out = conv_block(out)
#             intermediate_features_out.append(out)

#         # --- 3. 다중 레벨 특징 융합 ---
#         last_feature_map = intermediate_features_out[-1]
#         target_size = last_feature_map.shape[2:]

#         feature_maps_to_concat = []

#         # 3-1. 초기 입력 특징 리사이즈 및 추가
#         resized_initial_features = F.interpolate(features_for_processing, size=target_size, mode='trilinear', align_corners=False)
#         feature_maps_to_concat.append(resized_initial_features)

#         # 3-2. 중간 Conv 블록 특징들 리사이즈 및 추가
#         for feature_map in intermediate_features_out:
#             resized_map = F.interpolate(feature_map, size=target_size, mode='trilinear', align_corners=False)
#             feature_maps_to_concat.append(resized_map)

#         # 3-3. 모든 특징을 채널 방향으로 융합
#         cat_feature = torch.cat(feature_maps_to_concat, dim=1)

#         # --- 4. Logits 계산 ---
#         pooled_features = self.ranked_pooling(cat_feature)
#         logits = self.classifier(pooled_features)

#         # --- 5. CAM 생성 ---
#         cam = F.conv3d(
#             cat_feature,
#             self.classifier.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         )
        
#         activated_cam = self.cam_activation(cam)

#         return {
#             "logits": logits,
#             "cam": activated_cam
#         }


















class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)


# Convolution 사용 버전
# class CAMGenerator(nn.Module):
#     def __init__(self, input_hidden_size: int, num_classes: int, base_f: int = 128):
#         super().__init__()

#         # 1. Conv3D 블록
#         self.conv_blocks = nn.ModuleList([
#             Conv3DBlock(input_hidden_size, base_f),
#             Conv3DBlock(base_f, base_f * 2),
#             Conv3DBlock(base_f * 2, base_f * 4)
#         ])

#         # 그냥 Conv 통과한 결과물을 대상으로 진행하는 버전
#         final_conv_out_channels = base_f * 4  # 512
#         self.classifier = nn.Conv3d(final_conv_out_channels, num_classes, kernel_size=1, padding=0)

#         # 4. 순위 기반 풀링 모듈
#         self.ranked_pooling = RankedTopKPooling(k1_ratio=0.1, k2_ratio=1.0)

#         # 5. 시각화를 위한 CAM 활성화 함수
#         self.cam_activation = nn.ReLU(inplace=True)



#         # 다중 Feature Map 융합 버전
#         # 다중 레벨 특징 융합을 위한 채널 수 계산
#         # concatenated_channels = input_hidden_size + base_f + (base_f * 2) + (base_f * 4)

#         # 최종 분류기 (1x1x1 Conv3D로 변경)
#         # self.classifier = nn.Conv3d(concatenated_channels, num_classes, kernel_size=1, padding=0)


#     def forward(self, features: torch.Tensor):

#         # --- 1. 입력 준비 ---
#         batch_size, n_frames, seq_len, hidden_size = features.shape
#         patch_features = features[:, :, 1:, :]
#         h_w = int((seq_len - 1) ** 0.5)
#         patch_features_reshaped = patch_features.view(batch_size, n_frames, h_w, h_w, hidden_size)
#         features_for_processing = patch_features_reshaped.permute(0, 4, 1, 2, 3)



#         # 그냥 Conv 통과한 결과물을 대상으로 진행하는 버전
#         # Conv3D 블록들을 순차적으로 통과시켜 최종 특징맵을 얻음
#         # out = features_for_processing
#         # for conv_block in self.conv_blocks:
#         #     out = conv_block(out)

#         # # 최종 Conv3D 블록의 출력을 분류에 사용할 특징맵으로 지정
#         # final_feature_map = out

#         # # --- 3. 다중 레벨 특징 융합 과정 (제거됨) ---

#         # # --- 4. CAM 생성 및 Logits 계산 ---
#         # # 단순화된 최종 특징맵을 분류기에 바로 전달하여 CAM 생성
#         # cam = self.classifier(final_feature_map)

#         # # 생성된 CAM에 풀링을 적용하여 Logits 계산
#         # logits = self.ranked_pooling(cam)

#         # # 시각화를 위한 CAM 활성화
#         # activated_cam = self.cam_activation(cam)

#         # return {
#         #     "logits": logits,
#         #     "cam": activated_cam
#         # }




#         # 다중 Feature Map 융합 버전
#         # intermediate_features_out = []
#         # out = features_for_processing

#         # for conv_block in self.conv_blocks:
#         #     out = conv_block(out)
#         #     intermediate_features_out.append(out)

#         # # --- 3. 다중 레벨 특징 융합 ---
#         # last_feature_map = intermediate_features_out[-1]
#         # target_size = last_feature_map.shape[2:]

#         # feature_maps_to_concat = []

#         # # 3-1. 초기 입력 특징 리사이즈 및 추가
#         # resized_initial_features = F.interpolate(features_for_processing, size=target_size, mode='trilinear', align_corners=False)
#         # feature_maps_to_concat.append(resized_initial_features)

#         # # 3-2. 중간 Conv 블록 특징들 리사이즈 및 추가
#         # for feature_map in intermediate_features_out:
#         #     resized_map = F.interpolate(feature_map, size=target_size, mode='trilinear', align_corners=False)
#         #     feature_maps_to_concat.append(resized_map)

#         # # 3-3. 모든 특징을 채널 방향으로 융합
#         # cat_feature = torch.cat(feature_maps_to_concat, dim=1)

#         # # --- 4. CAM 생성 및 Logits 계산 (순서 변경) ---
#         # # 1x1x1 Conv를 통과시켜 CAM 생성
#         # # shape은 ([8, 10, 16, 7, 7]) --> 10개의 각 클래스에 대해, 16개 프레임의 어느 공간(7x7)이 얼마나 중요한가??
#         # cam = self.classifier(cat_feature)

#         # # 생성된 CAM에 풀링을 적용하여 Logits 계산
#         # # shape은 ([8, 10])
#         # logits = self.ranked_pooling(cam)

#         # # 시각화를 위한 CAM 활성화
#         # activated_cam = self.cam_activation(cam)

#         # return {
#         #     "logits": logits,
#         #     "cam": activated_cam
#         # }






#################################################################


class MLP3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                mlp_ratio: int = 4, dropout: float = 0.0):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_channels = int(in_channels * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Get original shape
        B, C, D, H, W = x.shape

        # Reshape for MLP: (B, C, D, H, W) -> (B, D*H*W, C)
        x_reshaped = x.flatten(2).permute(0, 2, 1)

        # Apply MLP
        x_mlp = self.mlp(x_reshaped)

        # Reshape back to 5D tensor: (B, D*H*W, C_out) -> (B, 
        # C_out, D*H*W) -> (B, C_out, D, H, W)
        x_out = x_mlp.permute(0, 2, 1).view(B, self.out_channels, D, H, W)

        return x_out


# MLP 사용 버전
class CAMGenerator(nn.Module):
    def __init__(self, input_hidden_size: int, num_classes: int, base_f: int = 128):
        super().__init__()

        # MLP 블록들은 그대로 사용
        self.mlp_blocks = nn.ModuleList([
            MLP3D(input_hidden_size, base_f),
            MLP3D(base_f, base_f * 2),
            MLP3D(base_f * 2, base_f * 4)
        ])

        concatenated_channels = input_hidden_size + base_f + (base_f * 2) + (base_f * 4)

        self.classifier = nn.Conv3d(concatenated_channels, num_classes, kernel_size=1, padding=0)

        self.ranked_pooling = RankedTopKPooling(k1_ratio=0.1, k2_ratio=1.0)

        self.cam_activation = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor):
        batch_size, n_frames, seq_len, hidden_size = features.shape
        patch_features = features[:, :, 1:, :]
        h_w = int((seq_len - 1) ** 0.5)
        patch_features_reshaped = patch_features.view(batch_size, n_frames, h_w, h_w, hidden_size)
        features_for_processing = patch_features_reshaped.permute(0, 4, 1, 2, 3)

        intermediate_features_out = []
        out = features_for_processing

        for mlp_block in self.mlp_blocks:
            out = mlp_block(out)
            intermediate_features_out.append(out)

        last_feature_map = intermediate_features_out[-1]
        target_size = last_feature_map.shape[2:]
        feature_maps_to_concat = []
        resized_initial_features = F.interpolate(features_for_processing, size=target_size, mode='trilinear', align_corners=False)
        feature_maps_to_concat.append(resized_initial_features)

        for feature_map in intermediate_features_out:
            resized_map = F.interpolate(feature_map, size=target_size, mode='trilinear', align_corners=False)
            feature_maps_to_concat.append(resized_map)

        # cat_feature shape: (B, C_concat, D, H, W)
        cat_feature = torch.cat(feature_maps_to_concat, dim=1)

        # --- Logits 및 CAM 계산 흐름 수정 ---
        
        # --- 4. CAM 생성 및 Logits 계산 (순서 변경) ---
        # 1x1x1 Conv를 통과시켜 CAM 생성
        cam = self.classifier(cat_feature)

        # 생성된 CAM에 풀링을 적용하여 Logits 계산
        logits = self.ranked_pooling(cam)

        # 시각화를 위한 CAM 활성화
        activated_cam = self.cam_activation(cam)

        return {
            "logits": logits,
            "cam": activated_cam
        }


#################################################################


# Hybrid 버전
# class CAMGenerator(nn.Module):
#     def __init__(self, input_hidden_size: int, num_classes: int, base_f: int = 128):
#         super().__init__()

#         self.feature_blocks = nn.ModuleList([
#             # 1x1 Conv로 ViT 특징을 먼저 처리
#             Conv3DBlock(input_hidden_size, base_f, kernel_size=1, padding=0), 
#             # 3x3 Conv로 로컬 패턴 학습
#             Conv3DBlock(base_f, base_f * 2, kernel_size=3, padding=1),
#             # 필요하다면 더 깊게
#             Conv3DBlock(base_f * 2, base_f * 4, kernel_size=3, padding=1)
#         ])

#         concatenated_channels = input_hidden_size + base_f + (base_f * 2) + (base_f * 4)

#         self.classifier = nn.Conv3d(concatenated_channels, num_classes, kernel_size=1, padding=0)

#         self.ranked_pooling = RankedTopKPooling(k1_ratio=0.05, k2_ratio=1.0)

#         self.cam_activation = nn.ReLU(inplace=True)

#     def forward(self, features: torch.Tensor):
#         batch_size, n_frames, seq_len, hidden_size = features.shape
#         patch_features = features[:, :, 1:, :]
#         h_w = int((seq_len - 1) ** 0.5)
#         patch_features_reshaped = patch_features.view(batch_size, n_frames, h_w, h_w, hidden_size)
#         features_for_processing = patch_features_reshaped.permute(0, 4, 1, 2, 3)

#         intermediate_features_out = []
#         out = features_for_processing

#         for conv_block in self.feature_blocks: # mlp_blocks 대신 feature_blocks 사용
#             out = conv_block(out)
#             intermediate_features_out.append(out)

#         last_feature_map = intermediate_features_out[-1]
#         target_size = last_feature_map.shape[2:]
#         feature_maps_to_concat = []
#         resized_initial_features = F.interpolate(features_for_processing, size=target_size, mode='trilinear', align_corners=False)
#         feature_maps_to_concat.append(resized_initial_features)

#         for feature_map in intermediate_features_out:
#             resized_map = F.interpolate(feature_map, size=target_size, mode='trilinear', align_corners=False)
#             feature_maps_to_concat.append(resized_map)

#         # cat_feature shape: (B, C_concat, D, H, W)
#         cat_feature = torch.cat(feature_maps_to_concat, dim=1)

#         # --- Logits 및 CAM 계산 흐름 수정 ---
        
#         # --- 4. CAM 생성 및 Logits 계산 (순서 변경) ---
#         # 1x1x1 Conv를 통과시켜 CAM 생성
#         cam = self.classifier(cat_feature)

#         # 생성된 CAM에 풀링을 적용하여 Logits 계산
#         logits = self.ranked_pooling(cam)

#         # 시각화를 위한 CAM 활성화
#         activated_cam = self.cam_activation(cam)

#         return {
#             "logits": logits,
#             "cam": activated_cam
#         }


#################################################################


# -----------------------------------------------------------------------------
# 1. Self-Attention 블록
# -----------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    """3D 공간/시간 데이터에 대한 Self-Attention을 수행하는 모듈"""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x의 입력 형태: (b, n, d) - b:배치, n:시퀀스 길이, d:차원
        # 3D 데이터를 1D 시퀀스로 변환
        b, c, depth, h, w = x.shape
        x_reshaped = rearrange(x, 'b c d h w -> b (d h w) c')

        # Q, K, V 생성
        qkv = self.to_qkv(x_reshaped).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Attention Score 계산
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # 최종 출력 계산
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        # 다시 원래의 3D 형태로 복원
        return rearrange(out, 'b (d h w) c -> b c d h w', d=depth, h=h, w=w)

# -----------------------------------------------------------------------------
# 2. 1x1 Conv와 Attention을 결합한 하이브리드 블록
# -----------------------------------------------------------------------------
class HybridBlock(nn.Module):
    """1x1 Conv -> Self-Attention -> 1x1 Conv 구조의 블록"""
    def __init__(self, in_channels, out_channels, heads=8):
        super().__init__()
        
        # 채널 차원을 조절하고 특징을 정제하는 1x1 Conv (MLP 역할)
        self.pre_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        # 공간/시간적 전역 관계를 학습하는 Attention
        self.attn = AttentionBlock(dim=out_channels, heads=heads)
        
        # Layer Normalization과 잔차 연결(Residual Connection)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # 피드포워드 네트워크 역할을 하는 1x1 Conv
        self.feed_forward = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(out_channels * 4, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # 1. 입력 특징을 1x1 Conv로 처리
        x = self.pre_conv(x)
        
        # 2. Attention 적용 (잔차 연결 포함)
        # LayerNorm을 위해 차원 순서 변경 (B, C, D, H, W) -> (B, D, H, W, C)
        x_permuted = x.permute(0, 2, 3, 4, 1)
        normed_x = self.norm1(x_permuted)
        normed_x = normed_x.permute(0, 4, 1, 2, 3) # 다시 원래대로
        x = self.attn(normed_x) + x
        
        # 3. FeedForward 적용 (잔차 연결 포함)
        x_permuted = x.permute(0, 2, 3, 4, 1)
        normed_x = self.norm2(x_permuted)
        normed_x = normed_x.permute(0, 4, 1, 2, 3)
        x = self.feed_forward(normed_x) + x
        
        return x


# class CAMGenerator(nn.Module):
#     def __init__(self, input_hidden_size: int, num_classes: int, base_f: int = 128):
#         super().__init__()

#         # MLP/CNN 블록 대신 새로운 하이브리드 블록을 사용
#         self.feature_blocks = nn.ModuleList([
#             HybridBlock(input_hidden_size, base_f),
#             HybridBlock(base_f, base_f * 2),
#             HybridBlock(base_f * 2, base_f * 4)
#         ])

#         # 이 부분은 이전 구조와 동일하게 특징들을 결합
#         concatenated_channels = input_hidden_size + base_f + (base_f * 2) + (base_f * 4)
#         self.classifier = nn.Conv3d(concatenated_channels, num_classes, kernel_size=1, padding=0)
#         self.cam_activation = nn.ReLU(inplace=True)
#         self.ranked_pooling = RankedTopKPooling(k1_ratio=0.05, k2_ratio=0.4)


#     def forward(self, features: torch.Tensor):
#         # ViT 출력(패치 임베딩)을 3D 그리드로 변환하는 과정 (이전과 동일)
#         batch_size, n_frames, seq_len, hidden_size = features.shape
#         patch_features = features[:, :, 1:, :]
#         h_w = int((seq_len - 1) ** 0.5)
#         patch_features_reshaped = patch_features.view(batch_size, n_frames, h_w, h_w, hidden_size)
#         features_for_processing = patch_features_reshaped.permute(0, 4, 1, 2, 3)

#         # 하이브리드 블록 통과 및 중간 특징 저장
#         intermediate_features_out = []
#         out = features_for_processing
#         for block in self.feature_blocks:
#             out = block(out)
#             intermediate_features_out.append(out)

#         # 모든 레벨의 특징 맵들을 결합 (이전과 동일)
#         last_feature_map = intermediate_features_out[-1]
#         target_size = last_feature_map.shape[2:]
        
#         feature_maps_to_concat = [F.interpolate(features_for_processing, size=target_size, mode='trilinear', align_corners=False)]
#         for feature_map in intermediate_features_out:
#             resized_map = F.interpolate(feature_map, size=target_size, mode='trilinear', align_corners=False)
#             feature_maps_to_concat.append(resized_map)

#         cat_feature = torch.cat(feature_maps_to_concat, dim=1)

#         # CAM 생성 및 Logits 계산
#         cam = self.classifier(cat_feature)
#         logits = self.ranked_pooling(cam).squeeze(-1).squeeze(-1).squeeze(-1) # (B, C)
#         activated_cam = self.cam_activation(cam)

#         return {
#             "logits": logits,
#             "cam": activated_cam
#         }





# def train_one_epoch_with_cam(video_model, dataloader, criterion, optimizer, device, epoch, output_dir):
#     video_model.train()
#     total_loss = 0.0
#     correct_predictions = 0
#     total_samples = 0

#     for batch_idx, (videos, sensors, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
#         videos = videos.to(device)
#         sensors = sensors.to(device)
#         labels = labels.to(device)
#         batch_size = videos.size(0)
        
#         optimizer.zero_grad()

#         # ==================================================================================
#         # 1단계: 단일 순전파로 모든 결과 얻기
#         # ==================================================================================
#         # ViTWithCAM 모델은 forward pass 한 번으로 필요한 모든 것을 반환합니다.
#         model_output = video_model(videos)
        
#         logits = model_output['logits'] # 최종 분류 예측
#         all_class_cam = model_output['cam'] # 모든 클래스에 대한 CAM (B, num_classes, T, H, W) --> ([8, 10, 16, 7, 7])
#         intermediate_features = model_output['intermediate_features'] # 마스킹 대상 특징

#         batch_size, n_frames, n_classes = logits.shape

#         # --- 해결을 위한 코드 ---

#         # 1. Logits의 차원을 [B, C, T] 형태로 변경합니다.
#         # nn.CrossEntropyLoss는 Class 차원이 두 번째(dim=1)에 올 것으로 기대합니다.
#         logits_permuted = logits.permute(0, 2, 1)

#         # 2. Labels의 형태를 [B, T]로 확장합니다.
#         # [B] -> [B, 1]로 차원을 늘린 뒤,
#         # T (프레임 수) 만큼 값을 복사하여 [B, T] 형태로 만듭니다.
#         labels_expanded = labels.unsqueeze(1).expand(batch_size, n_frames)

#         # 이제 올바른 차원의 텐서로 프레임별 손실을 계산합니다.
#         # PyTorch는 내부적으로 각 프레임의 손실을 계산한 뒤 평균을 냅니다.
#         classification_loss = criterion(logits_permuted, labels_expanded)

#         # ==================================================================================
#         # 2단계: 예측 클래스에 해당하는 CAM 선택
#         # ==================================================================================
#         # 모델의 예측 클래스를 기반으로 해당 클래스의 CAM을 마스킹에 사용합니다.
#         predicted_classes = logits.argmax(dim=1)
        
#         # torch.arange를 사용하여 각 배치 아이템의 인덱스를 생성하고,
#         # predicted_classes를 사용하여 해당 인덱스에서 원하는 클래스의 CAM을 선택합니다.
#         # 결과 shape: (B, T, H, W) -> ([8, 16, 7, 7])
#         cam_masks_tensor = all_class_cam[torch.arange(batch_size), predicted_classes]

#         # 수정 제안:
#         if batch_idx == 0:
#             # (A) 현재 배치에서 타겟 레이블의 인덱스를 찾음
#             # 여기 수정!!!!
#             target_indices = (labels == 0).nonzero(as_tuple=True)[0]

#             # (B) 조건에 따라 시각화할 샘플의 인덱스를 결정
#             if len(target_indices) > 0:
#                 # 타겟 레이블이 있으면: 첫 번째 인덱스 사용
#                 idx_to_visualize = target_indices[0].item()
#             else:
#                 # 타겟 레이블이 없으면: 랜덤 인덱스 사용
#                 batch_size = videos.size(0)
#                 idx_to_visualize = torch.randint(0, batch_size, (1,)).item()
            
#             # (C) 결정된 인덱스로 시각화할 데이터를 선택
#             video_to_viz = videos[idx_to_visualize].detach()
#             cam_to_viz = all_class_cam[idx_to_visualize].detach()
#             pred_to_viz = predicted_classes[idx_to_visualize].detach()

#             # 선택된 단일 샘플 데이터로 시각화 함수 호출
#             grid_image = visualize_cam_on_video_grid(
#                 video_tensor=video_to_viz,
#                 cam_tensor=cam_to_viz,
#                 predicted_class_indices=pred_to_viz
#             )

#             # 파일 이름은 에폭과 배치 번호를 포함하여 고유하게 만듭니다.
#             filename = os.path.join(output_dir, f"epoch_{epoch+1}_batch_{batch_idx}.png")
#             grid_image.save(filename)
#             print(f"CAM visualization saved to {filename}")

#             # wandb에 로깅합니다. step=epoch을 사용하여 에폭별로 이미지를 기록합니다.
#             # if grid_image:
#             #     wandb.log({
#             #         "Train/CAM_Visualization": wandb.Image(grid_image, caption=f"Epoch {epoch}")
#             #     }, step=epoch) # step 인자에 epoch을 직접 지정

#         # ==================================================================================
#         # 3단계: 어텐션 마스킹 및 최종 임베딩 생성 (기존 로직과 거의 동일)
#         # ==================================================================================
        
#         # (B, T, H, W) -> ([8, 16, 7, 7])
#         b, t, h, w = cam_masks_tensor.shape
        
#         # 3a. 어텐션 마스킹
#         # intermediate_features는 (B, T, 패치개수, Hidden) --> ([8, 16, 50, 768]) 형태
#         # 패치 개수는 CLS 토큰 1개 + 패치 토큰 49개 = 50개
#         # 여기서 CLS 토큰은 공간 정보가 없으므로 제외 --> 결국 ([8, 16, 49, 768]) 형태
#         patch_features = intermediate_features[:, :, 1:, :]  
        
#         patch_grid_size = int(np.sqrt(patch_features.shape[2])) # 7

#         # CAM을 (B, T, 패치H, 패치W) 로 리사이즈     
#         # CAMGenerator가 생성한 CAM(cam_masks_tensor)의 공간적 해상도(7x7)를, 마스킹할 대상인 patch_features의 공간적 해상도(패치 그리드 크기, 7x7)와 정확히 일치시킵니다.
#         # patch_features의 49개 패치는 사실 7x7 그리드 형태로 배열된 것이므로, 7x7 크기의 CAM을 각 패치 위치에 하나씩 정확하게 곱해주기 위함입니다.
#         # 이 코드에서는 이미 크기가 같지만, 다른 모델을 쓸 경우를 대비한 안전장치입니다
#         resized_cam = F.interpolate(cam_masks_tensor, size=(patch_grid_size, patch_grid_size), mode='bilinear', align_corners=False)
        
#         # 마스크를 패치 특징에 적용하기 위해 차원을 맞춥니다.
#         # resized_cam은 (8, 16, 7, 7) 형태이고, 이는 8개의 비디오, 각 16개의 프레임에 대해 7x7 그리드 형태의 중요도 맵을 의미함
#         # patch_features는 (8, 16, 49, 768) 형태이고, 8개 비디오, 각 16개 프레임에 대해, 49개의 패치 토큰이 각각 768차원의 특징 벡터를 가짐을 의미함
#         # 이 둘을 곱하려면 모양이 맞아야 한다!!
#         # 결국 view와 unsqueeze를 통해 (8, 16, 7, 7) -> (8, 16, 49, 1)로 변환하여 브로드캐스팅을 가능하게 합니다.
#         # 이러면 patch_features의 49개 패치와 일대일로 대응이 될 수 있음
#         resized_cam_flat = resized_cam.view(b, t, -1).unsqueeze(-1)
        
#         # 어텐션 마스킹
#         # masked_patch_features의 shape은 (B, T, 49, Hidden) --> ([8, 16, 49, 768])
#         masked_patch_features = patch_features * resized_cam_flat

#         # 시각화 로직 추가
#         # ==================================================================================
#         # ⭐️ [수정] 시각화 로직 추가 ⭐️
#         # ==================================================================================
#         # if batch_idx == 0:
#         #     # 시각화 함수를 호출하여 그리드 이미지를 생성합니다.
#         #     # 이 함수는 내부적으로 배치의 첫 번째 샘플만 사용합니다.
#         #     feature_image = visualize_features_on_video_grid(
#         #         video_tensor=videos.detach(), # 그래디언트 추적 불필요
#         #         feture_tensor=masked_patch_features.detach()
#         #     )

#         #     # wandb에 로깅합니다. step=epoch을 사용하여 에폭별로 이미지를 기록합니다.
#         #     if feature_image:
#         #         wandb.log({
#         #             "Train/Masked_Feature_Visualization": wandb.Image(feature_image, caption=f"Epoch {epoch}"),
#         #             "epoch": epoch
#         #         })

#         # 3b. 마스킹된 특징의 공간적, 시간적 통합
#         spatially_pooled_features = masked_patch_features.mean(dim=2) # (B, T, Hidden)
#         final_embedding = spatially_pooled_features.mean(dim=1) # (B, Hidden)

#         # ==================================================================================
#         # 4단계: 최종 손실 계산 및 학습
#         # ==================================================================================
#         # 현재는 분류 손실을 메인 손실로 사용합니다.
#         main_loss = classification_loss

#         main_loss.backward()
#         optimizer.step()
        
#         # 통계 기록
#         total_loss += main_loss.item() * videos.size(0)
#         total_samples += labels.size(0)
#         correct_predictions += (predicted_classes == labels).sum().item()

#     avg_loss = total_loss / total_samples
#     avg_acc = correct_predictions / total_samples
    
#     # wandb.log({
#     #     "Train/Loss": avg_loss,
#     #     "Train/Accuracy": avg_acc,
#     #     "epoch": epoch
#     # })
    
#     return avg_loss, avg_acc







def train_one_epoch_with_cam(video_model, dataloader, criterion, optimizer, device, epoch, output_dir):
    video_model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_frames = 0 # [수정] 샘플 수를 비디오가 아닌 프레임 기준으로 변경
    epoch_v_motions = [] 

    for batch_idx, (videos, sensors, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
        videos = videos.to(device)
        sensors = sensors.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 1단계: 단일 순전파로 모든 결과 얻기
        model_output = video_model(videos)
        logits = model_output['logits']              # (B, T, C)
        all_class_cam = model_output['cam']          # (B, T, C, H, W)
        intermediate_features = model_output['intermediate_features']

        batch_size, n_frames, n_classes = logits.shape

        # --- 프레임별 손실 계산 (기존 코드 유지 - 올바른 방식) ---
        logits_permuted = logits.permute(0, 2, 1)    # (B, C, T)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, n_frames) # (B, T)
        classification_loss = criterion(logits_permuted, labels_expanded)

        # ==================================================================================
        # 2단계: 예측 클래스에 해당하는 CAM 선택 (수정된 로직)
        # ==================================================================================
        # [수정] 각 프레임별 예측 클래스를 계산 (dim=2 사용)
        # 결과 shape: (B, T)
        predicted_classes = torch.argmax(logits, dim=2)

        # [수정] torch.gather를 사용하여 예측 클래스에 해당하는 CAM을 안전하게 선택
        # gather를 위해 cam과 predicted_classes의 차원을 조정합니다.
        # cam: (B, T, C, H, W) -> 그대로 사용
        # pred: (B, T) -> (B, T, 1, 1, 1)로 확장하여 인덱싱 준비
        pred_indices_expanded = predicted_classes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pred_indices_expanded = pred_indices_expanded.expand(-1, -1, -1, all_class_cam.shape[3], all_class_cam.shape[4])

        # all_class_cam의 클래스 차원(dim=2)에서 예측 인덱스에 해당하는 CAM을 수집
        # 결과 shape: (B, T, 1, H, W)
        gathered_cam = torch.gather(all_class_cam, 2, pred_indices_expanded)

        # 클래스 차원을 제거하여 최종 마스크 획득
        # 결과 shape: (B, T, H, W)
        cam_masks_tensor = gathered_cam.squeeze(2)

        # # ==================================================================================
        # # 시각화 로직 (첫 번째 배치에 대해서만 실행)
        # # ==================================================================================
        # if batch_idx == 0:
            
        #     # 여기 수정!!!
        #     # 시각화할 샘플 인덱스 결정 (기존 로직 유지 - 좋은 방식)
        #     target_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
        #     if len(target_indices) > 0:
        #         idx_to_visualize = target_indices[0].item()
        #     else:
        #         idx_to_visualize = torch.randint(0, batch_size, (1,)).item()

        #     # 결정된 인덱스로 시각화할 데이터 선택
        #     video_to_viz = videos[idx_to_visualize].detach()
        #     cam_to_viz = all_class_cam[idx_to_visualize].detach() # 모든 클래스 CAM 전달
        #     pred_to_viz = predicted_classes[idx_to_visualize].detach() # [수정] 올바르게 계산된 예측 전달

        #     # 시각화 함수 호출
        #     grid_image = visualize_cam_on_video_grid(
        #         video_tensor=video_to_viz,
        #         cam_tensor=cam_to_viz,
        #         predicted_class_indices=pred_to_viz
        #     )

        #     if grid_image:
        #         filename = os.path.join(output_dir, f"epoch_{epoch+1}_cam_visualization.png")
        #         grid_image.save(filename)
        #         print(f"CAM visualization saved to {filename}")

                # wandb.log({"Train/CAM_Visualization": wandb.Image(grid_image, caption=f"Epoch {epoch+1}")}, step=epoch)

        # ==================================================================================
        # 3단계: 어텐션 마스킹 및 최종 임베딩 생성 (기존 로직과 거의 동일)
        # ==================================================================================
        patch_features = intermediate_features[:, :, 1:, :]
        b, t, num_patches, hidden_dim = patch_features.shape
        patch_grid_size = int(np.sqrt(num_patches))

        resized_cam = F.interpolate(cam_masks_tensor, size=(patch_grid_size, patch_grid_size), mode='bilinear', align_corners=False)
        resized_cam_flat = resized_cam.view(b, t, -1).unsqueeze(-1)
        masked_patch_features = patch_features * resized_cam_flat # (B, T, num_patches, hidden_dim) --> (16, 16, 49, 768)


        # # ==================================================================================
        # # [추가] 마스킹된 피처맵 시각화 로직
        # # ==================================================================================
        # # 시각화할 샘플의 마스킹된 피처맵 선택 (CAM 시각화와 동일한 인덱스 사용)
        # features_to_viz = masked_patch_features[idx_to_visualize].detach()

        # # 1. 피처맵 가공: (T, num_patches, hidden_dim) -> (T, H_feat, W_feat)
        # # hidden_dim 차원에 대해 평균을 내어 차원 축소
        # feature_strengths = features_to_viz.mean(dim=-1) # -> (T, num_patches)

        # # 2D 그리드로 재구성
        # # patch_grid_size는 이전에 계산된 값을 사용해야 합니다. (예: 14)
        # # 만약 이전에 없다면, 여기서 다시 계산: patch_grid_size = int(np.sqrt(num_patches))
        # feature_map_2d = feature_strengths.view(
        #     -1, patch_grid_size, patch_grid_size
        # ) # -> (T, patch_grid_size, patch_grid_size)

        # # 2. 새로운 시각화 함수 호출
        # # 원본 비디오(video_to_viz)와 가공된 피처맵(feature_map_2d)을 전달
        # feature_grid_image = visualize_features_on_video_grid(
        #     video_tensor=video_to_viz,
        #     feature_map_tensor=feature_map_2d
        # )

        # # 3. 결과 저장 및 로깅
        # if feature_grid_image:
        #     feature_filename = os.path.join(output_dir, f"epoch_{epoch+1}_feature_visualization.png")
        #     feature_grid_image.save(feature_filename)
        #     print(f"Feature map visualization saved to {feature_filename}")


        # Motion Feature
        # ==================================================================================
        # [추가] 프레임 차분을 이용한 모션 벡터(v_motion) 생성
        # ==================================================================================
        # 1. 연속 프레임 특징 준비 (f_t, f_t+1)
        # masked_patch_features의 Shape: (B, T, num_patches, hidden_dim)
        features_t = masked_patch_features[:, :-1, :, :]
        features_t_plus_1 = masked_patch_features[:, 1:, :, :]

        # 2. 프레임 차분 계산 (d_t = f_t+1 - f_t)
        # frame_diffs Shape: (B, T-1, num_patches, hidden_dim)
        frame_diffs = features_t_plus_1 - features_t

        # 3. 시간 축 통합 (Temporal Average Pooling)
        # aggregated_diff_map Shape: (B, num_patches, hidden_dim)
        aggregated_diff_map = frame_diffs.mean(dim=1)

        # 4. 공간 축 통합 (Global Average Pooling)
        # v_motion Shape: (B, hidden_dim) --> (16, 768)
        v_motion = aggregated_diff_map.mean(dim=1)

        # v_motion 텐서를 CPU로 옮긴 후 NumPy 배열로 변환
        epoch_v_motions.append(v_motion.detach().cpu().numpy())

        # ==================================================================================
        # 4단계: 최종 손실 계산 및 학습
        # ==================================================================================
        main_loss = classification_loss
        main_loss.backward()
        optimizer.step()

        # --- 통계 기록 (수정된 로직) ---
        total_loss += main_loss.item() * batch_size # Loss는 배치 단위로 평균되므로 배치 크기를 곱함

        # [수정] 정확도는 프레임 단위로 계산
        correct_predictions += (predicted_classes == labels_expanded).sum().item()
        total_frames += (batch_size * n_frames)

    visualize_v_motion_tsne(epoch_v_motions, epoch, output_dir)

    # [수정] 평균 손실과 정확도 계산
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = correct_predictions / total_frames

    # wandb.log({"Train/Loss": avg_loss, "Train/Accuracy": avg_acc, "epoch": epoch})

    return avg_loss, avg_acc