import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from sklearn.manifold import TSNE


#################################################################


START_INDEX = 134+60
END_INDEX = 231
# START_INDEX = 1
# END_INDEX = 11

# ACTION_MERGE_LABELS = {
#         0: 'Door 1',
#         1: 'Door 2',
#         2: 'Fridge',
#         3: 'Dishwasher',
#         4: 'Drawer 1',
#         5: 'Drawer 2',
#         6: 'Drawer 3',
#         7: 'Clean Table',
#         8: 'Drink from Cup',
#         9: 'Toggle Switch'
#     }

ACTION_MERGE_LABELS = {
    "0": "Ktch_B1_Drawer",
    "1": "Ktch_B4_Cupboard",
    "2": "Ktch_Motion_1",
    "3": "Ktch_Motion_2",
    "4": "Ktch_T1_Cupboard",
    "5": "Ktch_T2_Cupboard",
    "6": "Ktch_T3_Cupboard",
    "7": "None Behavior",
    "8": "TP_L_Power"
}

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


#################################################################


# --- 6. t-SNE 시각화 함수 ---
def visualize_tsne_2d(embeddings, true_labels, pred_labels, prototypes=None, title="t-SNE Visualization 2D", num_classes=10):
    """2D t-SNE 결과를 시각화하고 Matplotlib Figure 객체를 반환"""
    label_names = [ACTION_MERGE_LABELS.get(i, f"Class_{i}") for i in range(num_classes)]
    
    # t-SNE를 실행하기에 샘플 수가 충분한지 확인
    if len(embeddings) <= 1:
        print(f"Warning: Cannot run t-SNE with {len(embeddings)} samples. Skipping visualization.")
        return plt.figure()  # 빈 Figure 객체 반환

    # 프로토타입과 임베딩을 함께 변환
    if prototypes is not None:
        combined_data = np.vstack([embeddings, prototypes])
    else:
        combined_data = embeddings
    nan_count = np.isnan(combined_data).sum()
    print(f"Total number of NaN values: {nan_count}")

    # NaN 값이 있는 행(샘플) 확인
    rows_with_nan = np.any(np.isnan(combined_data), axis=1)
    print(f"Rows containing NaN: \n{np.where(rows_with_nan)[0]}")
    # t-SNE 시각화 (2D와 3D 모두)

    perplexity_value = min(30, len(combined_data) - 1)
    if perplexity_value <= 0:
        perplexity_value = 1.0
        
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, metric="cosine")
    reduced_all = tsne.fit_transform(combined_data)
    
    reduced_embeddings = reduced_all[:len(embeddings)]
    if prototypes is not None:
        reduced_prototypes = reduced_all[len(embeddings):]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(title, fontsize=16)
    
    # 일관된 색상 매핑을 위한 색상 정의
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    # 실제 레이블 기준 시각화
    scatter1 = None
    for i in range(num_classes):
        mask = (true_labels == i)
        if mask.any():
            sc = ax1.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                    color=colors[i], label=label_names[i], alpha=0.7)
            if scatter1 is None:
                scatter1 = sc
    
    ax1.set_title("True Labels")

    # 예측된 클러스터 기준 시각화
    scatter2 = None
    for i in range(num_classes):
        mask = (pred_labels == i)
        if mask.any():
            sc = ax2.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                    color=colors[i], label=label_names[i], alpha=0.7)
            if scatter2 is None:
                scatter2 = sc
    
    # 프로토타입 시각화
    if prototypes is not None:
        proto_labels = np.arange(len(prototypes))
        ax1.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5)
        ax2.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5)
        
        # 프로토타입에 번호 추가
        for i in range(len(prototypes)):
            ax1.text(reduced_prototypes[i, 0] + 0.1, reduced_prototypes[i, 1] + 0.1, f'P{i}', fontsize=12, weight='bold')
            ax2.text(reduced_prototypes[i, 0] + 0.1, reduced_prototypes[i, 1] + 0.1, f'P{i}', fontsize=12, weight='bold')
        
        # 범례 수동 생성
        handles1 = []
        for i in range(num_classes):
            # 각 클래스마다 색상을 일관되게 설정
            handle = plt.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=label_names[i])
            handles1.append(handle)
        
        proto_handle = plt.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='Prototypes')
        handles1.append(proto_handle)
        ax1.legend(handles=handles1, labels=[h.get_label() for h in handles1])
        
        # 두 번째 그래프도 동일한 방식으로 범례 생성
        handles2 = []
        for i in range(num_classes):
            handle = plt.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=label_names[i])
            handles2.append(handle)
        
        handles2.append(proto_handle)
        ax2.legend(handles=handles2, labels=[h.get_label() for h in handles2])
    else:
        # 범례 수동 생성 (프로토타입 없음)
        handles1 = []
        handles2 = []
        for i in range(num_classes):
            handle1 = plt.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=label_names[i])
            handle2 = plt.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=label_names[i])
            handles1.append(handle1)
            handles2.append(handle2)
        
        ax1.legend(handles=handles1, labels=[h.get_label() for h in handles1])
        ax2.legend(handles=handles2, labels=[h.get_label() for h in handles2])
    
    return fig


#################################################################


def visualize_tsne_3d(embeddings, true_labels, pred_labels, prototypes=None, title="t-SNE Visualization 3D", num_classes=10):
    """3D t-SNE 결과를 시각화하고 Matplotlib Figure 객체를 반환"""
    label_names = [ACTION_MERGE_LABELS.get(i, f"Class_{i}") for i in range(num_classes)]
    
    # t-SNE를 실행하기에 샘플 수가 충분한지 확인
    if len(embeddings) <= 1:
        print(f"Warning: Cannot run t-SNE with {len(embeddings)} samples. Skipping visualization.")
        return plt.figure()  # 빈 Figure 객체 반환

    # 프로토타입과 임베딩을 함께 변환
    if prototypes is not None:
        combined_data = np.vstack([embeddings, prototypes])
    else:
        combined_data = embeddings

    perplexity_value = min(30, len(combined_data) - 1)
    if perplexity_value <= 0:
        perplexity_value = 1.0
        
    # 3차원 t-SNE 실행
    tsne = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, metric="cosine")
    reduced_all = tsne.fit_transform(combined_data)
    
    reduced_embeddings = reduced_all[:len(embeddings)]
    if prototypes is not None:
        reduced_prototypes = reduced_all[len(embeddings):]

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(title, fontsize=16)
    
    # 3D 서브플롯 생성
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 일관된 색상 매핑을 위한 색상 정의
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    # 실제 레이블 기준 시각화
    scatter1 = None
    for i in range(num_classes):
        mask = (true_labels == i)
        if mask.any():
            sc = ax1.scatter(
                reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], reduced_embeddings[mask, 2],
                color=colors[i], label=label_names[i], alpha=0.7, s=50
            )
            if scatter1 is None:
                scatter1 = sc
    ax1.set_title("True Labels")
    
    # 예측된 클러스터 기준 시각화
    scatter2 = None
    for i in range(num_classes):
        mask = (pred_labels == i)
        if mask.any():
            sc = ax2.scatter(
                reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], reduced_embeddings[mask, 2],
                color=colors[i], label=label_names[i], alpha=0.7, s=50
            )
            if scatter2 is None:
                scatter2 = sc
    ax2.set_title("Predicted Clusters")
    
    # 프로토타입 시각화
    if prototypes is not None:
        proto_labels = np.arange(len(prototypes))
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
            ax1.text(reduced_prototypes[i, 0], reduced_prototypes[i, 1], reduced_prototypes[i, 2], 
                    f'P{i}', fontsize=12, weight='bold')
            ax2.text(reduced_prototypes[i, 0], reduced_prototypes[i, 1], reduced_prototypes[i, 2], 
                    f'P{i}', fontsize=12, weight='bold')
    
    # 범례 수동 생성
    handles1 = []
    for i in range(num_classes):
        # 각 클래스마다 색상을 일관되게 설정
        handle = plt.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=label_names[i])
        handles1.append(handle)
    
    if prototypes is not None:
        proto_handle = plt.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='Prototypes')
        handles1.append(proto_handle)
        ax1.legend(handles=handles1, labels=[h.get_label() for h in handles1], loc='upper left')
    else:
        ax1.legend(handles=handles1, labels=[h.get_label() for h in handles1], loc='upper left')
    
    # 두 번째 그래프도 동일한 방식으로 범례 생성
    handles2 = []
    for i in range(num_classes):
        handle = plt.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=label_names[i])
        handles2.append(handle)
    
    if prototypes is not None:
        proto_handle = plt.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='Prototypes')
        handles2.append(proto_handle)
        ax2.legend(handles=handles2, labels=[h.get_label() for h in handles2], loc='upper left')
    else:
        ax2.legend(handles=handles2, labels=[h.get_label() for h in handles2], loc='upper left')
    
    # 축 라벨 설정
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_zlabel('Component 3')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')
    
    # 그래프 조절
    plt.tight_layout()
    
    return fig


#################################################################


def visualize_tsne(embeddings, true_labels, pred_labels, prototypes=None, title="t-SNE Visualization", num_classes=10):
    """2D와 3D t-SNE 시각화를 모두 수행하고 2D 결과를 반환"""
    # 2D 시각화
    fig_2d = visualize_tsne_2d(embeddings, true_labels, pred_labels, prototypes, title + " (2D)", num_classes)
    
    # 3D 시각화
    fig_3d = visualize_tsne_3d(embeddings, true_labels, pred_labels, prototypes, title + " (3D)", num_classes)
    
    # 기존 호환성을 위해 2D 그림 반환
    return fig_2d, fig_3d


#################################################################


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
    # column_names_path = "/home/jaemo/dataset_hwu_usp/extracted/hwu_usp_dataset/HWU-USP_v2/column_names.txt"
    
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

def visualize_sensor_name(top_indices, labels, id):
    for i in range(top_indices.shape[0]): # 배치 크기만큼 반복
        label = labels[i]
        # Top-K 인덱스들을 순회
        if id is not None:
            print(f"batch {i} id: {id[i]}")
            print(f"batch {i} label: {ACTION_MERGE_LABELS[label.item()]}")
            for index_tensor in top_indices[i]:
                index_val = index_tensor.item()
                sensor_name = get_sensor_name(index_val + START_INDEX + 1)
                print(f"  - Index: {index_val}, Name: {sensor_name}, {ACTION_MERGE_LABELS[label.item()]}")