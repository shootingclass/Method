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
from torchvision.transforms.functional import to_pil_image
from scipy.optimize import linear_sum_assignment


#################################################################


START_INDEX = 134+60
END_INDEX = 231
# START_INDEX = 1
# END_INDEX = 11

ACTION_MERGE_LABELS_OPPORTUNITY = {
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

ACTION_MERGE_LABELS_OPPORTUNITY_ALL = {0: 'Open Door 1',
    1: 'Open Door 2',
    2: 'Close Door 1',
    3: 'Close Door 2',
    4: 'Open Fridge',
    5: 'Close Fridge',
    6: 'Open Dishwasher',
    7: 'Close Dishwasher',
    8: 'Open Drawer 1',
    9: 'Close Drawer 1',
    10: 'Open Drawer 2',
    11: 'Close Drawer 2',
    12: 'Open Drawer 3',
    13: 'Close Drawer 3',
    14: 'Clean Table',
    15: 'Drink from Cup',
    16: 'Toggle Switch'}

ACTION_MERGE_LABELS_HWU_USP = {
    0: "Ktch_B4_Cupboard",
    1: "Ktch_Motion_1",
    2: "Ktch_Motion_2",
    3: "Ktch_T1_Cupboard",
    4: "Ktch_T2_Cupboard",
    5: "Ktch_T3_Cupboard",
    6: "None Behavior"
}

#################################################################


# --- 헝가리안 매칭을 통한 클러스터-라벨 매핑 ---
def compute_hungarian_matching(pred_labels, true_labels, num_clusters):
    """클러스터 ID와 실제 레이블 간의 최적 매핑을 찾아 정확도를 계산"""
    print(f"start Computing Hungarian matching... {len(pred_labels)}")
    cost_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for i in range(len(pred_labels)):
        cost_matrix[pred_labels[i], true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapped_preds = np.zeros_like(pred_labels)
    mapping = {i: j for i, j in zip(row_ind, col_ind)}
    for i, j in mapping.items():
        mapped_preds[pred_labels == i] = j
    accuracy = np.mean(mapped_preds == true_labels)
    print("Accuracy: ", accuracy, "Mapping: ", mapping)
    return accuracy, mapping

def denormalize(tensor):
    """텐서를 정규화 해제합니다."""
    # 텐서를 복제하여 원본이 변경되지 않도록 합니다.
        # 프레임 전처리(Transform) 정의
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]

    # 🌟🌟🌟 수정된 부분: 리스트를 텐서로 변환! 🌟🌟🌟
    # device=tensor.device를 추가하여 GPU/CPU 문제를 방지합니다.
    mean = torch.tensor(MEAN, device=tensor.device)
    std = torch.tensor(STD, device=tensor.device)

    # 이제 mean과 std는 텐서이므로 .view()를 사용할 수 있습니다.
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    # 역정규화 계산 및 0~1 범위 고정
    denormalized_tensor = (tensor * std + mean).clamp(0, 1)
    
    return denormalized_tensor

def visualize_cropped_tensor(cropped_video_tensor: torch.Tensor, title: str = "Cropped Video Frame"):
    """
    크롭된 비디오 텐서를 올바르게 시각화하고 Matplotlib Figure 객체를 반환합니다.
    """
    if cropped_video_tensor.ndim != 4:
        raise ValueError(f"Input tensor must be 4D (C, T, H, W). Got {cropped_video_tensor.ndim}D.")
    
    frame_index_to_show = cropped_video_tensor.shape[1] // 2 
    cropped_frame_tensor = cropped_video_tensor[:, frame_index_to_show, :, :]

    # ‼️‼️‼️ 중요: 정규화 해제 단계 추가 ‼️‼️‼️
    # 데이터셋을 만들 때 사용했던 mean과 std 값을 여기에 정확히 입력해야 합니다.
    # 예시 값 (ImageNet 기준):
    # MEAN = [0.485, 0.456, 0.406]
    # STD = [0.229, 0.224, 0.225]class MethodDataModule(pl.LightningDataModule):
        
    # # CPU로 이동시킨 후 정규화 해제
    denormalized_frame = denormalize(cropped_frame_tensor.cpu())
    
    # # 값 범위를 [0, 1]로 안전하게 클리핑
    denormalized_frame = torch.clamp(denormalized_frame, 0, 1)
    # Tensor를 PIL Image로 변환
    cropped_frame_pil = to_pil_image(denormalized_frame)
    # cropped_frame_pil = to_pil_image(cropped_frame_tensor.cpu())
    
    # Matplotlib으로 시각화
    fig, ax = plt.subplots(figsize=(8, 8)) # fig와 ax를 함께 받습니다.
    ax.imshow(cropped_frame_pil)
    ax.set_title(f"{title} (Frame {frame_index_to_show}) - Shape: {cropped_frame_pil.size[1]}x{cropped_frame_pil.size[0]}")
    ax.axis('off')

    # ‼️‼️‼️ 중요: plt가 아닌 fig 객체 반환 ‼️‼️‼️
    return fig

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
def visualize_tsne_2d(embeddings, true_labels, pred_labels=None, prototypes=None,
                      title="t-SNE Visualization 2D", num_classes=10, dataset_name="Opportunity++"):
    """pred_labels가 None이면 True Labels만 시각화"""
    
    # --- label dictionary 선택 ---
    if dataset_name == "Opportunity++":
        ACTION_MERGE_LABELS = ACTION_MERGE_LABELS_OPPORTUNITY if num_classes <= 10 else ACTION_MERGE_LABELS_OPPORTUNITY_ALL
    elif dataset_name == "HWU-USP":
        ACTION_MERGE_LABELS = ACTION_MERGE_LABELS_HWU_USP
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    label_names = [ACTION_MERGE_LABELS.get(i, f"Class_{i}") for i in range(num_classes)]

    # --- t-SNE 실행 준비 ---
    if len(embeddings) <= 1:
        print(f"Warning: Cannot run t-SNE with {len(embeddings)} samples.")
        return plt.figure()

    combined_data = np.vstack([embeddings, prototypes]) if prototypes is not None else embeddings
    perplexity_value = max(1, min(30, len(combined_data) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, metric="cosine")
    reduced_all = tsne.fit_transform(combined_data)
    reduced_embeddings = reduced_all[:len(embeddings)]
    reduced_prototypes = reduced_all[len(embeddings):] if prototypes is not None else None

    # --- 컬러맵 설정 ---
    cmap = plt.cm.get_cmap('tab20', num_classes)
    colors = cmap(np.linspace(0, 1, num_classes))

    # --- pred_labels 존재 여부에 따라 subplot 구성 ---
    if pred_labels is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(title + " (True Labels Only)", fontsize=16)

        for i in range(num_classes):
            mask = (true_labels == i)
            if mask.any():
                ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                           color=colors[i], label=label_names[i], alpha=0.7)
        ax.set_title("True Labels")

        if prototypes is not None:
            ax.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1],
                       c='black', marker='X', s=200, edgecolor='white', linewidth=1.5, label='Prototypes')
            for i in range(len(prototypes)):
                ax.text(reduced_prototypes[i, 0]+0.1, reduced_prototypes[i, 1]+0.1, f'P{i}', fontsize=10, weight='bold')

        ax.legend()
        return fig

    # --- pred_labels가 있는 경우: 기존처럼 두 subplot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(num_classes):
        mask = (true_labels == i)
        if mask.any():
            ax1.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                        color=colors[i], label=label_names[i], alpha=0.7)
    ax1.set_title("True Labels")

    for i in range(num_classes):
        mask = (pred_labels == i)
        if mask.any():
            ax2.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                        color=colors[i], label=label_names[i], alpha=0.7)
    ax2.set_title("Predicted Clusters")

    if prototypes is not None:
        ax1.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1],
                    c='black', marker='X', s=200, edgecolor='white', linewidth=1.5, label='Prototypes')
        ax2.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1],
                    c='black', marker='X', s=200, edgecolor='white', linewidth=1.5, label='Prototypes')
        for i in range(len(prototypes)):
            ax1.text(reduced_prototypes[i, 0]+0.1, reduced_prototypes[i, 1]+0.1, f'P{i}', fontsize=10, weight='bold')
            ax2.text(reduced_prototypes[i, 0]+0.1, reduced_prototypes[i, 1]+0.1, f'P{i}', fontsize=10, weight='bold')

    ax1.legend()
    ax2.legend()
    return fig


#################################################################


def visualize_tsne_3d(embeddings, true_labels, pred_labels=None, prototypes=None,
                      title="t-SNE Visualization 3D", num_classes=10, dataset_name="Opportunity++"):
    """pred_labels가 None이면 True Labels만 시각화"""
    
    # --- label dictionary 선택 ---
    if dataset_name == "Opportunity++":
        ACTION_MERGE_LABELS = ACTION_MERGE_LABELS_OPPORTUNITY if num_classes <= 10 else ACTION_MERGE_LABELS_OPPORTUNITY_ALL
    elif dataset_name == "HWU-USP":
        ACTION_MERGE_LABELS = ACTION_MERGE_LABELS_HWU_USP
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    label_names = [ACTION_MERGE_LABELS.get(i, f"Class_{i}") for i in range(num_classes)]

    if len(embeddings) <= 1:
        print(f"Warning: Cannot run t-SNE with {len(embeddings)} samples.")
        return plt.figure()

    combined_data = np.vstack([embeddings, prototypes]) if prototypes is not None else embeddings
    perplexity_value = max(1, min(30, len(combined_data) - 1))
    tsne = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, metric="cosine")
    reduced_all = tsne.fit_transform(combined_data)
    reduced_embeddings = reduced_all[:len(embeddings)]
    reduced_prototypes = reduced_all[len(embeddings):] if prototypes is not None else None

    cmap = plt.cm.get_cmap('tab20', num_classes)
    colors = cmap(np.linspace(0, 1, num_classes))

    # --- pred_labels가 없는 경우: 단일 3D plot ---
    if pred_labels is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title + " (True Labels Only)", fontsize=16)

        for i in range(num_classes):
            mask = (true_labels == i)
            if mask.any():
                ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], reduced_embeddings[mask, 2],
                           color=colors[i], label=label_names[i], alpha=0.7)
        if prototypes is not None:
            ax.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], reduced_prototypes[:, 2],
                       c='black', marker='X', s=200, edgecolor='white', linewidth=1.5, label='Prototypes')

        ax.legend()
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        return fig

    # --- pred_labels 있는 경우: 기존 2-subplot 버전 ---
    fig = plt.figure(figsize=(22, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.suptitle(title, fontsize=16)

    for i in range(num_classes):
        mask = (true_labels == i)
        if mask.any():
            ax1.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], reduced_embeddings[mask, 2],
                        color=colors[i], label=label_names[i], alpha=0.7)
    ax1.set_title("True Labels")

    for i in range(num_classes):
        mask = (pred_labels == i)
        if mask.any():
            ax2.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], reduced_embeddings[mask, 2],
                        color=colors[i], label=label_names[i], alpha=0.7)
    ax2.set_title("Predicted Clusters")

    if prototypes is not None:
        ax1.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], reduced_prototypes[:, 2],
                    c='black', marker='X', s=200, edgecolor='white', linewidth=1.5, label='Prototypes')
        ax2.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], reduced_prototypes[:, 2],
                    c='black', marker='X', s=200, edgecolor='white', linewidth=1.5, label='Prototypes')

    ax1.legend()
    ax2.legend()
    return fig



#################################################################


def visualize_tsne(embeddings, true_labels, pred_labels, prototypes=None, title="t-SNE Visualization", num_classes=10, dataset_name="Opportunity++"):
    """2D와 3D t-SNE 시각화를 모두 수행하고 2D 결과를 반환"""
    # 2D 시각화
    fig_2d = visualize_tsne_2d(embeddings, true_labels, pred_labels, prototypes, title + " (2D)", num_classes, dataset_name)
    
    # 3D 시각화
    fig_3d = visualize_tsne_3d(embeddings, true_labels, pred_labels, prototypes, title + " (3D)", num_classes, dataset_name)
    
    # 기존 호환성을 위해 2D 그림 반환
    return fig_2d, fig_3d


#################################################################


def get_sensor_name(sensor_index, dataset_name="Opportunity++"):
    """
    sensor_index에 해당하는 센서 이름을 반환합니다.
    
    Args:
        sensor_index: 센서 인덱스 (1-based index)
    
    Returns:
        str: 센서 이름 문자열, 해당 인덱스가 없으면 "Unknown Sensor"
    """
    # 파일 경로 설정
    if dataset_name == "Opportunity++":
        column_names_path = "/mnt/hdd4tb/junho/Opportunity++/data/column_names.txt"
    elif dataset_name == "HWU-USP":
        column_names_path = "/mnt/hdd4tb/junho/dataset_hwu_usp/extracted/hwu_usp_dataset/HWU-USP_v2/column_names.txt"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
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

def visualize_sensor_name(top_indices, labels, id, dataset_name="Opportunity++"):
    if dataset_name == "Opportunity++":
        ACTION_MERGE_LABELS = ACTION_MERGE_LABELS_OPPORTUNITY
    elif dataset_name == "HWU-USP":
        ACTION_MERGE_LABELS = ACTION_MERGE_LABELS_HWU_USP
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
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


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import wandb

def visualize_joint_space(
    z_video_np, 
    z_sensor_np, 
    labels, 
    title="Joint Z-space", 
    max_samples=2000, 
    log_to_wandb=True
):
    # 1️⃣ 샘플 제한
    n = min(len(z_video_np), max_samples)
    idx = np.random.choice(len(z_video_np), n, replace=False)
    z_video_np = z_video_np[idx]
    z_sensor_np = z_sensor_np[idx]
    labels = labels[idx]

    # 2️⃣ 결합
    all_embs = np.concatenate([z_video_np, z_sensor_np], axis=0)
    all_labels = np.concatenate([labels, labels], axis=0)
    modalities = np.array([0]*len(z_video_np) + [1]*len(z_sensor_np))

    # 3️⃣ 2D t-SNE
    tsne_2d = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto")
    emb_2d = tsne_2d.fit_transform(all_embs)

    plt.figure(figsize=(8, 6))
    scatter_v = plt.scatter(
        emb_2d[modalities==0, 0], emb_2d[modalities==0, 1],
        c=all_labels[modalities==0], cmap="tab10", s=25, alpha=0.7, label="Video"
    )
    scatter_s = plt.scatter(
        emb_2d[modalities==1, 0], emb_2d[modalities==1, 1],
        c=all_labels[modalities==1], cmap="tab10", s=25, marker="x", alpha=0.7, label="Sensor"
    )
    plt.legend()
    plt.title(f"{title} (2D)")
    plt.tight_layout()
    plt.show()

    if log_to_wandb:
        wandb.log({f"{title}_2D": wandb.Image(plt)})
    plt.close()

    # 4️⃣ 3D t-SNE
    tsne_3d = TSNE(n_components=3, perplexity=30, init="random", learning_rate="auto")
    emb_3d = tsne_3d.fit_transform(all_embs)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    p1 = ax.scatter(
        emb_3d[modalities==0,0], emb_3d[modalities==0,1], emb_3d[modalities==0,2],
        c=all_labels[modalities==0], cmap="tab10", s=30, alpha=0.7, label="Video"
    )
    p2 = ax.scatter(
        emb_3d[modalities==1,0], emb_3d[modalities==1,1], emb_3d[modalities==1,2],
        c=all_labels[modalities==1], cmap="tab10", s=30, marker="x", alpha=0.7, label="Sensor"
    )
    ax.set_title(f"{title} (3D)")
    ax.legend()

    plt.tight_layout()
    if log_to_wandb:
        wandb.log({f"{title}_3D": wandb.Image(fig)})
    plt.show()
    plt.close(fig)

    return emb_2d, emb_3d

import numpy as np

def compute_alignment_score(z_video_np, z_sensor_np, labels_np):
    unique_labels = np.unique(labels_np)
    dists = []
    for lab in unique_labels:
        v_center = z_video_np[labels_np==lab].mean(axis=0)
        s_center = z_sensor_np[labels_np==lab].mean(axis=0)
        dist = np.linalg.norm(v_center - s_center)
        dists.append(dist)
    mean_dist = np.mean(dists)
    print(f"Alignment Score (lower is better): {mean_dist:.4f}")
    return mean_dist

import torch
import torch.nn.functional as F

def cross_modal_retrieval(z_video_np, z_sensor_np):
    v = F.normalize(torch.tensor(z_video_np), dim=1)
    s = F.normalize(torch.tensor(z_sensor_np), dim=1)
    sim = v @ s.T
    top1 = sim.argmax(dim=1).cpu().numpy()
    acc = np.mean(np.arange(len(v)) == top1)
    print(f"Cross-modal retrieval top-1 acc: {acc:.4f}")
    return acc

