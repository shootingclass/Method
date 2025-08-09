import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List
import torchvision.transforms.functional as TF


#################################################################


def _select_cam_for_prediction(
    cam_tensor_for_clip: torch.Tensor,
    predicted_class_idx: int
) -> torch.Tensor:
    """
    전체 클래스에 대한 CAM 텐서에서, 예측된 클래스에 해당하는 CAM만 선택합니다.

    Args:
        cam_tensor_for_clip (torch.Tensor): 단일 비디오 클립에 대한 CAM 텐서.
            - Shape: (Num_Classes, T, 7, 7)
        predicted_class_idx (int): 예측된 클래스의 인덱스.

    Returns:
        torch.Tensor: 선택된 클래스의 CAM.
            - Shape: (T, 7, 7)
    """
    return cam_tensor_for_clip[predicted_class_idx]


#################################################################


def _superimpose_heatmap_on_image(
    frame_tensor: torch.Tensor,
    heatmap_tensor: torch.Tensor,
    alpha: float = 0.5,
    colormap_name: str = 'jet'
) -> 'Image.Image':
    """
    단일 프레임 이미지 위에 단일 히트맵을 겹쳐서 PIL 이미지로 반환합니다.

    Args:
        frame_tensor (torch.Tensor): 원본 프레임 텐서.
            - Shape: (C, H, W)
        heatmap_tensor (torch.Tensor): 7x7 크기의 CAM 텐서.
            - Shape: (7, 7)
        alpha (float): 히트맵 투명도.
        colormap_name (str): 사용할 matplotlib 컬러맵 이름.

    Returns:
        PIL.Image.Image: 원본 프레임과 히트맵이 합성된 이미지.
    """
    # 1. 텐서를 CPU로 이동하고 NumPy 배열로 변환
    frame_np = frame_tensor.cpu().permute(1, 2, 0).numpy()

    # 2. 원본 프레임 정규화 (0-1 범위로) 및 8-bit 정수형으로 변환
    frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-6)
    frame_uint8 = (frame_np * 255).astype(np.uint8)
    
    # 3. 히트맵을 원본 프레임 크기로 업샘플링
    h, w, _ = frame_uint8.shape
    heatmap_resized = F.interpolate(
        heatmap_tensor.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).squeeze().detach().cpu().numpy()

    # 4. 히트맵 정규화 (0-1 범위로)
    heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-6)

    # 5. 컬러맵 적용하여 히트맵을 RGB 이미지로 변환
    colormap = cm.get_cmap(colormap_name)
    colored_heatmap = colormap(heatmap_normalized)[:, :, :3]  # Alpha 채널 제외
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

    # 6. PIL 이미지로 변환하고 합성
    frame_pil = Image.fromarray(frame_uint8)
    heatmap_pil = Image.fromarray(colored_heatmap_uint8)
    
    overlayed_image = Image.blend(frame_pil, heatmap_pil, alpha=alpha)

    return overlayed_image


#################################################################


def _create_image_grid(
    image_list: List[Image.Image],
    grid_cols: int
) -> 'Image.Image':
    """
    PIL 이미지 리스트를 받아서 하나의 그리드 이미지로 합칩니다.

    Args:
        image_list (list): PIL.Image 객체들의 리스트.
        grid_cols (int): 그리드의 열 수.

    Returns:
        PIL.Image.Image: 합쳐진 그리드 이미지.
    """
    if not image_list:
        return None
        
    w, h = image_list[0].size
    grid_rows = (len(image_list) + grid_cols - 1) // grid_cols
    
    grid_img = Image.new('RGB', (grid_cols * w, grid_rows * h))
    
    for i, img in enumerate(image_list):
        row = i // grid_cols
        col = i % grid_cols
        grid_img.paste(img, (col * w, row * h))
        
    return grid_img


#################################################################


def visualize_cam_on_video_grid(
    video_tensor: torch.Tensor,
    cam_tensor: torch.Tensor,
    predicted_class_indices: torch.Tensor,
    max_frames: int = 16,
    grid_cols: int = 4,
    heatmap_alpha: float = 0.5
) -> 'Image.Image':
    """
    비디오 텐서와 CAM 텐서를 받아, 각 프레임별 예측에 해당하는 CAM을 원본 프레임에
    오버레이한 그리드 이미지를 생성합니다.
    배치(batch) 데이터가 들어올 경우, 첫 번째 샘플만 사용합니다.

    Args:
        video_tensor (torch.Tensor): 원본 비디오 프레임 텐서.
            - Shape: (B, T, C, H, W) 또는 (T, C, H, W)
        cam_tensor (torch.Tensor): 모델이 생성한 전체 클래스에 대한 CAM 텐서.
            - Shape: (B, Num_Classes, T, 7, 7) 또는 (Num_Classes, T, 7, 7)
        predicted_class_indices (torch.Tensor): 각 프레임에 대해 예측된 클래스
    인덱스.
            - Shape: (B, T) 또는 (T,)
        max_frames (int): 시각화할 최대 프레임 수.
        grid_cols (int): 그리드 이미지의 열(column) 수.
        heatmap_alpha (float): 원본 이미지 위에 겹칠 히트맵의 투명도.

    Returns:
        PIL.Image.Image: 모든 시각화 결과가 포함된 하나의 그리드 이미지.
    """
    # 1. 입력 텐서 차원 처리 (배치 유무 확인 및 첫 번째 샘플 선택)
    if video_tensor.dim() == 5:  # 배치가 있는 경우 (B, T, C, H, W)
        video_clip = video_tensor[0]
        cam_clip = cam_tensor[0]  # Shape: (Num_Classes, T, 7, 7)
        preds_for_clip = predicted_class_indices[0]  # Shape: (T,)
    else:  # 단일 데이터인 경우 (T, C, H, W)
        video_clip = video_tensor
        cam_clip = cam_tensor
        preds_for_clip = predicted_class_indices

    # 2. 시각화할 프레임 인덱스 결정
    num_frames = video_clip.shape[0]
    if num_frames > max_frames:
        # 전체 프레임에서 max_frames 개수만큼 균일하게 샘플링
        indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(num_frames)

    video_frames_to_viz = video_clip[indices]

    # 3. 각 프레임에 대해 오버레이 이미지 생성 (수정된 핵심 로직)
    overlayed_images = []
    for i, frame_idx in enumerate(indices):
        # 현재 프레임(시각화 대상)
        frame_tensor = video_frames_to_viz[i]

        # 현재 프레임에 해당하는 예측 클래스 인덱스
        frame_pred_idx = preds_for_clip[frame_idx].item()

        # 현재 프레임의 예측 클래스에 해당하는 CAM 선택
        # cam_clip: (Num_Classes, T, 7, 7) -> cam_map: (7, 7)
        cam_map_for_frame = cam_clip[frame_idx, frame_pred_idx]

        # 히트맵 오버레이
        overlay = _superimpose_heatmap_on_image(
            frame_tensor=frame_tensor,
            heatmap_tensor=cam_map_for_frame,
            alpha=heatmap_alpha
        )
        overlayed_images.append(overlay)

    # 4. 이미지 그리드 생성
    grid = _create_image_grid(overlayed_images, grid_cols)

    return grid


#################################################################


def visualize_features_on_video_grid(
    video_tensor: torch.Tensor,
    feature_tensor: torch.Tensor,
    max_frames: int = 16,
    grid_cols: int = 4,
    heatmap_alpha: float = 0.5
) -> 'Image.Image':
    """
    비디오 텐서와 (마스킹된) 특징 텐서를 받아, 특징 벡터의 크기를 히트맵으로 시각화하여
    원본 프레임에 오버레이한 그리드 이미지를 생성합니다.
    배치(batch) 데이터가 들어올 경우, 첫 번째 샘플만 사용합니다.

    Args:
        video_tensor (torch.Tensor): 원본 비디오 프레임 텐서.
            - Shape: (B, T, C, H, W) 또는 (T, C, H, W)
        feature_tensor (torch.Tensor): 시각화할 패치 특징 텐서.
            - Shape: (B, T, Num_Patches, Hidden_Size) 또는 (T, Num_Patches, Hidden_Size)
        max_frames (int): 시각화할 최대 프레임 수.
        grid_cols (int): 그리드 이미지의 열(column) 수.
        heatmap_alpha (float): 원본 이미지 위에 겹칠 히트맵의 투명도.

    Returns:
        PIL.Image.Image: 모든 시각화 결과가 포함된 하나의 그리드 이미지.
    """
    # 1. 입력 텐서 차원 처리 (배치 유무 확인)
    if video_tensor.dim() == 5: # (B, T, C, H, W)
        video_clip = video_tensor[0]
        feature_clip = feature_tensor[0]
    else: # (T, C, H, W)
        video_clip = video_tensor
        feature_clip = feature_tensor

    # 2. 시각화할 프레임 선택
    num_frames = video_clip.shape[0]
    if num_frames > max_frames:
        indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(num_frames)

    video_frames_to_viz = video_clip[indices]
    features_to_viz = feature_clip[indices] # Shape: (max_frames, Num_Patches, Hidden_Size)

    # 3. 특징 텐서를 히트맵으로 변환 (핵심 로직)
    # 3a. 각 특징 벡터의 L2-norm을 계산하여 특징의 강도를 구합니다.
    # Shape: (max_frames, Num_Patches)
    feature_magnitudes = torch.norm(features_to_viz, p=2, dim=-1)

    # 3b. 패치 그리드 크기를 계산하고 2D 히트맵으로 재구성합니다.
    num_patches = feature_magnitudes.shape[1]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"The number of patches ({num_patches}) is not a perfect square.")

    # Shape: (max_frames, grid_size, grid_size) -> (16, 7, 7)
    heatmaps = feature_magnitudes.view(-1, grid_size, grid_size)

    # 4. 각 프레임에 대해 오버레이 이미지 생성
    overlayed_images = []
    for frame, heatmap in zip(video_frames_to_viz, heatmaps):
        overlay = _superimpose_heatmap_on_image(
            frame_tensor=frame,
            heatmap_tensor=heatmap,
            alpha=heatmap_alpha
        )
        overlayed_images.append(overlay)

    # 5. 이미지 그리드 생성
    grid = _create_image_grid(overlayed_images, grid_cols)

    return grid


#################################################################


def visualize_features_on_video_grid(video_tensor, feature_map_tensor, grid_size=None):

    if video_tensor.dim() != 4 or feature_map_tensor.dim() != 3:
        print("Error: Incorrect tensor dimensions.")
        return None

    if video_tensor.shape[0] != feature_map_tensor.shape[0]:
        print("Error: Mismatch in the number of frames (T).")
        return None

    # Ensure tensors are on CPU and detached from the computation graph
    video_tensor = video_tensor.detach().cpu()
    feature_map_tensor = feature_map_tensor.detach().cpu()

    n_frames, _, H, W = video_tensor.shape

    # --- Un-normalize video tensor for visualization ---
    # Assumes standard ImageNet normalization. Adjust if yours is different.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    # Apply to the whole batch of frames at once for efficiency
    video_tensor = video_tensor * std + mean
    video_tensor = torch.clamp(video_tensor, 0, 1)

    grid_cell_images = []

    for t in range(n_frames):
        # --- 1. Prepare the original video frame ---
        frame_pil = TF.to_pil_image(video_tensor[t])

        # --- 2. Prepare the feature map as a heatmap ---
        feature_map = feature_map_tensor[t]

        # Normalize the feature map to the [0, 1] range for the colormap
        fmin, fmax = feature_map.min(), feature_map.max()
        if fmax > fmin:
            feature_map = (feature_map - fmin) / (fmax - fmin)
        feature_map_np = feature_map.numpy()

        # Apply a colormap (e.g., 'viridis') and convert to a PIL image
        heatmap_np = plt.get_cmap('viridis')(feature_map_np)[:, :, :3]  # Drop the alpha channel
        heatmap_pil = Image.fromarray((heatmap_np * 255).astype(np.uint8))

        # Resize heatmap to match the original frame's dimensions
        heatmap_resized = heatmap_pil.resize(frame_pil.size, Image.Resampling.BILINEAR)

        # --- 3. Combine frame and heatmap side-by-side ---
        combined_pil = Image.new('RGB', (W * 2, H))
        combined_pil.paste(frame_pil, (0, 0))
        combined_pil.paste(heatmap_resized, (W, 0))

        # Add a label for the frame number for clarity
        draw = ImageDraw.Draw(combined_pil)
        draw.text((5, 5), f"Frame {t}", fill="white")

        grid_cell_images.append(combined_pil)

    if not grid_cell_images:
        return None

    # --- 4. Arrange all combined images into a single grid ---
    if grid_size is None:
        # Calculate a grid size that is as close to square as possible
        cols = int(np.ceil(np.sqrt(len(grid_cell_images))))
        rows = int(np.ceil(len(grid_cell_images) / cols))
    else:
        rows, cols = grid_size

    cell_w, cell_h = grid_cell_images[0].width, grid_cell_images[0].height
    final_grid_image = Image.new('RGB', (cols * cell_w, rows * cell_h))

    for i, img in enumerate(grid_cell_images):
        row_idx = i // cols
        col_idx = i % cols
        final_grid_image.paste(img, (col_idx * cell_w, row_idx * cell_h))

    return final_grid_image

