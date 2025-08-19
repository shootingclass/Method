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
from visualization import save_video_grid


####################################################################


def calculate_sensor_stats(dataset):
    """
    Calculates mean and std for each sensor channel across the entire dataset.
    """
    all_sensor_data = []
    print("Calculating sensor statistics...")
    
    for i in tqdm(range(len(dataset)), desc="Collecting sensor data"):
        _, sensor_data, _ = dataset[i]        
        all_sensor_data.append(sensor_data)
    
    concatenated_data = np.concatenate(all_sensor_data, axis=1)
    
    mean = np.mean(concatenated_data, axis=1)
    std = np.std(concatenated_data, axis=1)
    
    print("Calculation complete.")
    return {'mean': mean, 'std': std}


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
        
        # [추가] 2단계(어텐션 브릿지)의 출력 V'을 받아옵니다.
        # 이 변수는 3단계(모션 스트림)의 입력으로 사용될 예정입니다.
        transformed_video = model_output['transformed_video']

        # 시각화
        # ==================================================================================
        # [추가] 첫 번째 배치에 대한 transformed_video 시각화
        # ==================================================================================
        if batch_idx == 0:
            
            # 저장 경로를 epoch별로 다르게 설정합니다.
            vis_output_path = os.path.join(output_dir, "transformed_video", f"epoch_{epoch}.png")

            # 시각화 함수를 호출합니다.
            save_video_grid(transformed_video, vis_output_path)
        # ==================================================================================

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

    # [수정] 평균 손실과 정확도 계산
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = correct_predictions / total_frames

    # wandb.log({"Train/Loss": avg_loss, "Train/Accuracy": avg_acc, "epoch": epoch})

    return avg_loss, avg_acc