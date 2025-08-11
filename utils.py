import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm # tqdm 라이브러리 임포트 추가
import wandb
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# model.py에 저장된 모델 클래스를 임포트합니다.
from model import Clip4ClipVisionModel
from visualization import visualize_cam_on_video_grid, visualize_features_on_video_grid
from clustering_model import *

####################################################################


def calculate_sensor_stats(dataset):
    """
    Calculates mean and std for each sensor channel across the entire dataset.
    Uses pandas approach for consistency with z_score_channel_stats.py
    """
    import pandas as pd
    
    print("Calculating sensor statistics (pandas method)...")
    
    # 데이터셋에서 센서 데이터 추출하여 변환할 리스트
    all_samples = []
    
    for i in tqdm(range(len(dataset)), desc="Collecting sensor data"):
        _, sensor_data, _ = dataset[i]
        # sensor_data 형태: (센서 수, 시간 스텝)
        
        # 데이터를 전치하여 시간 스텝이 행이 되도록 함
        transposed_data = sensor_data.T  # 형태: (시간 스텝, 센서 수)
        
        # 각 시간 스텝을 개별 샘플로 취급
        for t in range(transposed_data.shape[0]):
            all_samples.append(transposed_data[t])
    
    # 모든 샘플을 행으로 포함하는 데이터프레임 생성
    # 각 열은 하나의 센서 채널에 해당
    df = pd.DataFrame(all_samples)
    
    print(f"데이터프레임 형태: {df.shape} (샘플 수, 센서 수)")
    
    # 판다스의 mean, std 함수를 사용하여 각 센서(열)별 통계 계산
    means = df.mean().values
    stds = df.std().values
    
    print("Calculation complete.")
    return {'mean': means, 'std': stds}


#################################################################


def save_stats(stats, path):
    """
    Saves the calculated statistics to a .npy file.
    """
    np.save(path, stats)
    print(f"Sensor statistics saved to {path}")


#################################################################


def load_stats(path):
    """
    Loads statistics from a .npy file.
    """
    stats = np.load(path, allow_pickle=True).item()
    print(f"Sensor statistics loaded from {path}")
    return stats


#################################################################


def train_one_epoch_with_cam(video_model, clustering_model, dataloader, criterion, optimizer, device, epoch, output_dir, args):
    video_model.train()
    clustering_model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_frames = 0 # [수정] 샘플 수를 비디오가 아닌 프레임 기준으로 변경
    

    for batch_idx, (videos, sensors, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
        videos = videos.to(device)
        sensors = sensors.to(device)
        labels = labels.to(device)
        # print("sensors", sensors)
        # print("labels", labels)
        # assert False


        optimizer.zero_grad()
        # 0. 클러스터링 모델 순전파 및 손실 계산 (비지도 학습)
        num_sensors = sensors.shape[1]
        rule_based_feature = get_representative_sensor_feature(sensors, labels, num_sensors, args.top_k)
        scores_cluster, final_feature_cluster, alpha = clustering_model(sensors, rule_based_feature)
        scores_sk = sinkhorn_knopp(scores_cluster, args.temperature, args.sk_iterations, device)
        # wandb.log({"train_alpha": alpha})
    
        with torch.no_grad():
            pseudo_labels = torch.argmax(scores_sk, dim=1)
            
        mse_loss = F.mse_loss(final_feature_cluster, clustering_model.prototypes[pseudo_labels])
        
        prototypes = clustering_model.prototypes
        p1 = prototypes.unsqueeze(1)
        p2 = prototypes.unsqueeze(0)
        mse_matrix = F.mse_loss(p1, p2, reduction='none').mean(dim=2)
        n_proto = args.num_classes
        diversity_loss = - (mse_matrix.sum()) / (n_proto * (n_proto - 1))
        
        loss_cluster = mse_loss + diversity_loss

        if epoch < args.threshold_epoch:
            # 9-epoch까지는 클러스터링 모델만 학습
            loss_cluster.backward()
            optimizer.step()
            total_loss += loss_cluster.item()
            continue
        # 일단 rule-based feature 사용 (almost 0.7 accuracy)
        labels = pseudo_labels

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

        # ==================================================================================
        # 시각화 로직 (첫 번째 배치에 대해서만 실행)
        # ==================================================================================
        if batch_idx == 0:
            
            # 여기 수정!!!
            # 시각화할 샘플 인덱스 결정 (기존 로직 유지 - 좋은 방식)
            target_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            if len(target_indices) > 0:
                idx_to_visualize = target_indices[0].item()
            else:
                idx_to_visualize = torch.randint(0, batch_size, (1,)).item()

            # 결정된 인덱스로 시각화할 데이터 선택
            video_to_viz = videos[idx_to_visualize].detach()
            cam_to_viz = all_class_cam[idx_to_visualize].detach() # 모든 클래스 CAM 전달
            pred_to_viz = predicted_classes[idx_to_visualize].detach() # [수정] 올바르게 계산된 예측 전달

            # 시각화 함수 호출
            grid_image = visualize_cam_on_video_grid(
                video_tensor=video_to_viz,
                cam_tensor=cam_to_viz,
                predicted_class_indices=pred_to_viz
            )

            if grid_image:
                filename = os.path.join(output_dir, f"epoch_{epoch+1}_cam_visualization.png")
                grid_image.save(filename)
                print(f"CAM visualization saved to {filename}")

                # wandb.log({"Train/CAM_Visualization": wandb.Image(grid_image, caption=f"Epoch {epoch+1}")}, step=epoch)

        # ==================================================================================
        # 3단계: 어텐션 마스킹 및 최종 임베딩 생성 (기존 로직과 거의 동일)
        # ==================================================================================
        patch_features = intermediate_features[:, :, 1:, :]
        b, t, num_patches, hidden_dim = patch_features.shape
        patch_grid_size = int(np.sqrt(num_patches))

        resized_cam = F.interpolate(cam_masks_tensor, size=(patch_grid_size, patch_grid_size), mode='bilinear', align_corners=False)
        resized_cam_flat = resized_cam.view(b, t, -1).unsqueeze(-1)
        masked_patch_features = patch_features * resized_cam_flat # (B, T, num_patches, hidden_dim)


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


        # 비디오 전체를 대표하는 최종 특징 벡터를 만들어야 함
        # spatially_pooled_features = masked_patch_features.mean(dim=2)
        # final_embedding = spatially_pooled_features.mean(dim=1)

        # ==================================================================================
        # 4단계: 최종 손실 계산 및 학습
        # ==================================================================================
        main_loss = classification_loss + loss_cluster
        main_loss.backward()
        optimizer.step()

        # --- 통계 기록 (수정된 로직) ---
        total_loss += main_loss.item() * batch_size # Loss는 배치 단위로 평균되므로 배치 크기를 곱함

        # [수정] 정확도는 프레임 단위로 계산
        correct_predictions += (predicted_classes == labels_expanded).sum().item()
        total_frames += (batch_size * n_frames)

    # [수정] 평균 손실과 정확도 계산
    if total_frames == 0:
        return total_loss, 0
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = correct_predictions / total_frames

    # wandb.log({"Train/Loss": avg_loss, "Train/Accuracy": avg_acc, "epoch": epoch})

    return avg_loss, avg_acc

def validation_one_epoch_with_cam(video_model, clustering_model, dataloader, criterion, device, epoch, output_dir, args):
    video_model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_frames = 0

    evaluate(clustering_model, dataloader, device, args, epoch, stage="val")
