import numpy as np
from tqdm import tqdm
import torch

####################################################################


def calculate_sensor_stats(dataset):
    """
    Calculates mean and std for each sensor channel across the entire dataset.
    """
    all_sensor_data = []
    print("Calculating sensor statistics...")
    
    for i in tqdm(range(len(dataset)), desc="Collecting sensor data"):
        _, sensor_data, _, _ = dataset[i]        
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

# --- 1. 센서 데이터 증강 (Data Augmentation) ---
def time_warp(x, sigma=0.2, num_knots=4):
    
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