import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List
import torchvision.transforms.functional as TF
from sklearn.manifold import TSNE
import os
# clustering model
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#################################################################
# clustering model

START_INDEX = 134
END_INDEX = 231

ACTION_MERGE_LABELS = {
        0: 'Door 1',
        1: 'Door 2',
        2: 'Fridge',
        3: 'Dishwasher',
        4: 'Drawer 1',
        5: 'Drawer 2',
        6: 'Drawer 3',
        7: 'Clean Table',
        8: 'Drink from Cup',
        9: 'Toggle Switch'
    }

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


def save_video_grid(video_tensor: torch.Tensor, output_path: str, nrow: int = None):
    """
    비디오 텐서로부터 프레임 그리드 이미지를 저장합니다.

    Args:
        video_tensor (torch.Tensor): (B, T, C, H, W) 형태의 비디오 텐서. 배치의 첫 번째 비디오(B=0)를 시각화합니다.
        output_path (str): 결과 이미지 그리드를 저장할 경로.
        nrow (int, optional): 그리드의 각 행에 표시할 이미지 수. None이면 모든 프레임(T)을 한 줄로 표시합니다. 기본값은 None.
    """
    # 저장할 디렉토리가 없으면 생성
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 배치에서 첫 번째 비디오를 선택 (T, C, H, W)
    video_to_show = video_tensor[0].detach().cpu()

    # nrow가 지정되지 않으면, 프레임 수를 행의 수로 설정하여 한 줄로 만듦
    if nrow is None:
        nrow = video_to_show.shape[0]

    # 텐서 값을 [0, 1] 범위로 클램핑하여 시각화에 적합하게 만듦
    # 참고: 만약 텐서가 [-1, 1] 범위로 정규화되었다면,
    # video_to_show = (video_to_show + 1) / 2 와 같은 코드가 필요할 수
    # 있습니다.
    video_to_show = video_to_show.clamp(0, 1)

    # 프레임들로 이미지 그리드 생성
    grid = torchvision.utils.make_grid(video_to_show, nrow=nrow, padding=2, normalize=False)

    # 텐서 그리드를 PIL 이미지로 변환
    # (C, H, W) -> (H, W, C) 차원 변경 후, [0, 255] 범위의 uint8 타입으로
    # 변환
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_img = Image.fromarray((grid_np * 255).astype(np.uint8))

    # 이미지 저장
    grid_img.save(output_path)
    print(f"Transformed video visualization saved to {output_path}")


def visualize_tsne_3D(embeddings, true_labels, pred_labels, prototypes=None, title="t-SNE Visualization", mapping=False):
    """
    t-SNE 결과를 3D로 시각화하고 Matplotlib Figure 객체를 반환합니다.
    """
    n_samples = prototypes.shape[0]
    label_names = [f"Class_{i}" for i in range(n_samples)]
    
    if n_samples <= 1:
        print(f"Warning: Cannot run t-SNE with {n_samples} samples.")
        return plt.figure()

    perplexity_value = min(30.0, float(n_samples - 1))
    if perplexity_value <= 0: perplexity_value = 1.0

    print(f"Running 3D t-SNE with {n_samples} samples and perplexity={perplexity_value:.1f}")
    
    # --- ✨ 핵심 수정: n_components=3 으로 변경 ---
    tsne = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, metric="cosine")
    
    if prototypes is not None:
        combined_data = np.vstack([embeddings, prototypes])
        reduced_data = tsne.fit_transform(combined_data)
        reduced_embeddings = reduced_data[:-len(prototypes)]
        reduced_prototypes = reduced_data[-len(prototypes):]
    else:
        reduced_embeddings = tsne.fit_transform(embeddings)
        reduced_prototypes = None

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(title, fontsize=16)

    # --- ✨ 핵심 수정: subplot을 3D로 설정 ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # 실제 레이블 기준 시각화
    scatter1 = ax1.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
        c=true_labels, cmap="tab10", alpha=0.7
    )
    ax1.set_title("True Labels")
    legend1_handles, _ = scatter1.legend_elements(num=n_samples)
    ax1.legend(legend1_handles, label_names)

    # 예측된 클러스터 기준 시각화
    scatter2 = ax2.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
        c=pred_labels, cmap="tab10", alpha=0.7
    )
    if mapping:
        ax2.set_title("Predicted Clusters (Mapped)")
    else:
        ax2.set_title("Predicted Clusters")
    
    if reduced_prototypes is not None:
        proto_labels = np.arange(len(prototypes))
        # 두 subplot에 모두 프로토타입을 표시
        ax1.scatter(
            reduced_prototypes[:, 0], reduced_prototypes[:, 1], reduced_prototypes[:, 2],
            c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5
        )
        ax2.scatter(
            reduced_prototypes[:, 0], reduced_prototypes[:, 1], reduced_prototypes[:, 2],
            c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5
        )
        
        # 프로토타입에 번호 추가
        for i in range(len(prototypes)):
            ax1.text(reduced_prototypes[i, 0], reduced_prototypes[i, 1], reduced_prototypes[i, 2], f'P{i}', fontsize=12, weight='bold')
            ax2.text(reduced_prototypes[i, 0], reduced_prototypes[i, 1], reduced_prototypes[i, 2], f'P{i}', fontsize=12, weight='bold')
    
    legend2_handles, _ = scatter2.legend_elements(num=n_samples)
    ax2.legend(legend2_handles, label_names)
        
    return fig
# LinearProbingEvaluator 클래스를 아래 코드로 교체하세요.
def visualize_tsne(embeddings, true_labels, pred_labels, title, prototypes=None, num_classes=7, mapping=False):
        """t-SNE 결과를 시각화하고 Matplotlib Figure 객체를 반환. 프로토타입도 함께 시각화 가능."""
        fig = visualize_tsne_3D(embeddings, true_labels, pred_labels, prototypes=prototypes, mapping=mapping)
        wandb.log({"t-SNE Visualization_3d": wandb.Image(fig, caption=title)})

        label_names = [f"Class_{i}" for i in range(num_classes)]
        assert num_classes == len(prototypes), f"num_classes must be equal to the number of prototypes, now num_classes: {num_classes}, len(prototypes): {len(prototypes)}"
        # --- ✨ 핵심 수정: t-SNE를 실행하기에 샘플 수가 충분한지 확인 ---
        if len(embeddings) <= 1:
            print(f"Warning: Cannot run t-SNE with {len(embeddings)} samples. Skipping visualization.")
            return plt.figure() # 빈 Figure 객체 반환

        # --- ✨ 프로토타입과 임베딩을 함께 변환하기 위해 결합 ---
        if prototypes is not None:
            combined_data = np.vstack([embeddings, prototypes])
        else:
            combined_data = embeddings

        # Perplexity는 샘플 수보다 작아야 함
        perplexity_value = min(30, len(combined_data) - 1)
        if perplexity_value <= 0: # 이중 안전장치
            perplexity_value = 1.0
        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, n_iter=300, metric="cosine")
        reduced_all = tsne.fit_transform(combined_data)
        
        reduced_embeddings = reduced_all[:len(embeddings)]
        if prototypes is not None:
            reduced_prototypes = reduced_all[len(embeddings):]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
        fig.suptitle(title, fontsize=16)
        
        # 실제 레이블 기준 시각화
        scatter1 = ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=true_labels, cmap="tab10", alpha=0.7)
        ax1.set_title("True Labels")

        # 예측된 클러스터 기준 시각화
        scatter2 = ax2.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=pred_labels, cmap="tab10", alpha=0.7)
        if mapping:
            ax2.set_title("Predicted Clusters (Mapped)")
        else:
            ax2.set_title("Predicted Clusters")
        
        # --- ✨ 프로토타입 시각화 추가 ---
        if prototypes is not None:
            proto_labels = np.arange(len(prototypes))
            # 두 subplot에 모두 프로토타입을 표시
            ax1.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5)
            ax2.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5)
            
            # 프로토타입에 번호 추가
            for i in range(len(prototypes)):
                ax1.text(reduced_prototypes[i, 0] + 0.1, reduced_prototypes[i, 1] + 0.1, f'P{i}', fontsize=12, weight='bold')
                ax2.text(reduced_prototypes[i, 0] + 0.1, reduced_prototypes[i, 1] + 0.1, f'P{i}', fontsize=12, weight='bold')
            
            # 범례 업데이트
            handles1 = scatter1.legend_elements(num=num_classes)[0]
            proto_handle = plt.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='Prototypes')
            handles1.append(proto_handle)
            ax1.legend(handles=handles1, labels=label_names + ['Prototypes'])

            print("\n--- Legend Debugging Info ---")
            # 실제로 pred_labels에 어떤 값들이 들어있는지 확인
            unique_preds = np.unique(pred_labels)
            print(f"Unique predicted labels in data: {unique_preds}")
            print(f"Number of unique predicted labels: {len(unique_preds)}")

            # legend_elements가 생성하는 핸들의 실제 개수 확인
            handles_check = scatter2.legend_elements(num=num_classes)[0]
            print(f"Number of handles generated by legend_elements: {len(handles_check)}")
            print(f"Number of labels provided: {len(label_names) + 1}")
            print("---------------------------\n")
            handles2 = scatter2.legend_elements(num=num_classes)[0]
            handles2.append(proto_handle)
            print('handles2: ', handles2)
            print('label_names: ', label_names)
            ax2.legend(handles=handles2, labels=label_names + ['Prototypes'])
        else:
            assert False, "prototypes is not None"
            ax1.legend(handles=scatter1.legend_elements(num=num_classes)[0], labels=label_names)
            ax2.legend(handles=scatter2.legend_elements(num=num_classes)[0], labels=label_names)
        
        return fig
   



def get_sensor_name(sensor_index):
    """
    sensor_index에 해당하는 센서 이름을 반환합니다.
    
    Args:
        sensor_index: 센서 인덱스 (1-based index)
    
    Returns:
        str: 센서 이름 문자열, 해당 인덱스가 없으면 "Unknown Sensor"
    """
    # 파일 경로 설정
    column_names_path = "/mnt/hdd4tb/junho/Opportunity++/data/column_names.txt"
    
    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(column_names_path):
            return f"Unknown Sensor (Index: {sensor_index})"
        
        # 파일 읽기
        with open(column_names_path, 'r') as f:
            lines = f.readlines()
        
        # 지정된 인덱스 찾기
        for line in lines:
            # Column: {index} {description} 형식 찾기
            if line.strip().startswith(f"Column: {sensor_index} "):
                # 센서 설명 추출
                sensor_description = line.strip()[len(f"Column: {sensor_index} "):]
                
                # 센서 이름과 타입 파싱 (예: "Accelerometer RKN^ accX")
                parts = sensor_description.split(';')[0].strip().split()
                if len(parts) >= 2:
                    sensor_type = parts[0]  # "Accelerometer"
                    sensor_location = parts[1]  # "RKN^"
                    sensor_axis = " ".join(parts[2:])  # "accX"
                    return f"{sensor_type} {sensor_location} {sensor_axis}"
                else:
                    return sensor_description
                
        # 인덱스가 없으면
        return f"Unknown Sensor (Index: {sensor_index})"
    
    except Exception as e:
        return f"Error reading sensor name: {str(e)}"