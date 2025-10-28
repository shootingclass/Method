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



# --- viz_motion.py 같은 곳에 두고 import 해도 되고, 그냥 파일 하단에 둬도 OK ---
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _norm01(x: torch.Tensor):
    x = x - x.min()
    return x / (x.max() + 1e-6)

def show_motion_recon_overlay(video, motion_target, motion_recon, idx=0, title="Motion Recon"):
    """
    video:         [B,T,C,H,W]
    motion_target: [B,1,h',w'] or [B,1,H,W]
    motion_recon:  [B,1,h',w']
    idx 번째 샘플을 '마지막 프레임' 위에 heatmap으로 오버레이해서 시각화.
    """
    assert video.ndim == 5 and motion_recon.ndim == 4 and motion_target.ndim == 4
    B, T, C, H, W = video.shape
    v_last = video[idx, -1].permute(1,2,0).detach().cpu().float()  # [H,W,C]
    v_last = _norm01(v_last)

    tgt = motion_target[idx, 0].detach().cpu().float()  # [h',w'] or [H,W]
    rec = motion_recon[idx, 0].detach().cpu().float()   # [h',w']

    # 시각화는 원본 프레임 크기(H,W)로 올려서 오버레이
    if tgt.shape != (H, W):
        tgt = F.interpolate(tgt[None,None], size=(H, W), mode="bilinear", align_corners=False)[0,0]
    if rec.shape != (H, W):
        rec = F.interpolate(rec[None,None], size=(H, W), mode="bilinear", align_corners=False)[0,0]
    err = (tgt - rec).abs()

    tgt = _norm01(tgt); rec = _norm01(rec); err = _norm01(err)

    fig, axes = plt.subplots(1,4, figsize=(18,5))
    axes[0].imshow(v_last);                     axes[0].set_title("Last Frame");     axes[0].axis("off")
    axes[1].imshow(v_last); im1 = axes[1].imshow(tgt, cmap="magma",   alpha=0.6)
    axes[1].set_title("Target (diff)");         axes[1].axis("off");   fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].imshow(v_last); im2 = axes[2].imshow(rec, cmap="magma",   alpha=0.6)
    axes[2].set_title("Reconstruction");        axes[2].axis("off");   fig.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[3].imshow(v_last); im3 = axes[3].imshow(err, cmap="inferno", alpha=0.6)
    axes[3].set_title("|Target−Recon|");        axes[3].axis("off");   fig.colorbar(im3, ax=axes[3], fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    return fig

import torch
import matplotlib.pyplot as plt
import wandb

def log_video_recon_grid(videos, video_recon, logger, step, max_n=4):
    """
    원본 vs 복원 비디오를 side-by-side로 시각화 (WandB logging용)
    Args:
        videos: [B, T, C, H, W]
        video_recon: [B, 3, T, H, W] or [B, T, 3, H, W]
    """
    B, T, C, H, W = videos.shape
    max_n = min(B, max_n)

    # 복원 비디오 차원 정렬
    if video_recon.shape[1] == 3 and video_recon.shape[2] == T:
        recon = video_recon
    elif video_recon.shape[2] == 3:  # [B, T, C, H, W]
        recon = video_recon.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unexpected video_recon shape: {video_recon.shape}")

    fig, axes = plt.subplots(max_n, T, figsize=(T * 2, max_n * 2))
    if max_n == 1:
        axes = [axes]

    for i in range(max_n):
        for t in range(T):
            orig = videos[i, t].permute(1, 2, 0).cpu().numpy()
            recn = recon[i, :, t].permute(1, 2, 0).cpu().detach().numpy()
            concat = torch.tensor(
                torch.clip(torch.from_numpy(
                    torch.cat((orig, recn), axis=0)
                ), 0, 1)
            )
            axes[i][t].imshow(concat)
            axes[i][t].axis("off")

    plt.tight_layout()
    logger.experiment.log({"video_recon_grid": wandb.Image(plt, caption=f"Step {step}")})
    plt.close()

import torch
import wandb
import torchvision
import numpy as np
import tempfile
import os

def log_video_recon_gif(videos, video_recon, logger, step, max_n=2, fps=4):
    """
    원본과 복원 비디오를 mp4/GIF 형태로 WandB에 함께 로깅.
    Args:
        videos: [B, T, C, H, W]
        video_recon: [B, C, T, H, W] or [B, T, C, H, W]
    """
    B, T, C, H, W = videos.shape
    max_n = min(B, max_n)

    # 정렬 (모델에 따라 [B,C,T,H,W] or [B,T,C,H,W])
    if video_recon.shape[1] == 3 and video_recon.shape[2] == T:
        recon = video_recon
    elif video_recon.shape[2] == 3:
        recon = video_recon.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unexpected shape: {video_recon.shape}")

    # normalize (0~1)
    videos = torch.clamp(videos, 0, 1)
    recon = torch.clamp(recon, 0, 1)

    # 각 샘플별로 원본/복원 합치기
    combined = []
    for i in range(max_n):
        orig_seq = videos[i].permute(1, 0, 2, 3)     # [T, C, H, W]
        recon_seq = recon[i].permute(1, 0, 2, 3)
        # 위-아래로 concat
        both = torch.cat([orig_seq, recon_seq], dim=2)  # H doubled
        combined.append(both)
    combined = torch.stack(combined)  # [B, T, C, H*2, W]

    # WandB Video expects [B, T, C, H, W] in [0,255]
    combined_np = (combined * 255).cpu().byte().numpy()

    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, f"recon_step{step}.mp4")

    # torchvision.utils.save_video 로 저장
    torchvision.io.write_video(video_path, combined_np[0].transpose(0, 2, 3, 1), fps=fps)

    # wandb video log
    logger.experiment.log({
        "video_reconstruction": wandb.Video(video_path, fps=fps, caption=f"Step {step}")
    })
