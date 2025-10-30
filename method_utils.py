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
import torchvision.utils as vutils
import numpy as np
import wandb
import imageio
import cv2

@torch.no_grad()
def log_video_recon_grid(videos, recons, logger, step, max_n=4, title="video_rec"):
    """
    Log a grid comparing original vs reconstructed videos.
    Handles numpy/tensor inputs and shape mismatches robustly.
    """
    try:
        # --- 타입 보정 ---
        if isinstance(videos, np.ndarray):
            videos = torch.from_numpy(videos)
        if isinstance(recons, np.ndarray):
            recons = torch.from_numpy(recons)

        videos = videos.detach().cpu()
        recons = recons.detach().cpu()

        # --- 배치 크기 제한 ---
        B = min(videos.size(0), max_n)
        videos = videos[:B]
        recons = recons[:B]

        # --- 크기 정렬: [B, T, C, H, W] or [B, C, T, H, W] 지원 ---
        if videos.dim() == 5 and videos.shape[2] in [1, 3]:
            pass
        elif videos.dim() == 5 and videos.shape[1] in [1, 3]:
            videos = videos.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Unexpected video shape {videos.shape}")

        if recons.dim() == 5 and recons.shape[2] in [1, 3]:
            pass
        elif recons.dim() == 5 and recons.shape[1] in [1, 3]:
            recons = recons.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Unexpected recon shape {recons.shape}")

        # --- Flatten time dimension ---
        videos_flat = videos.reshape(-1, *videos.shape[2:])   # [B*T, C, H, W]
        recons_flat = recons.reshape(-1, *recons.shape[2:])   # [B*T, C, H, W]

        # --- Channel 보정 ---
        if videos_flat.dim() == 3:
            videos_flat = videos_flat.unsqueeze(1)
        if recons_flat.dim() == 3:
            recons_flat = recons_flat.unsqueeze(1)

        # --- 크기 mismatch → resize recon ---
        if videos_flat.shape[-2:] != recons_flat.shape[-2:]:
            H, W = videos_flat.shape[-2:]
            recons_flat = torch.nn.functional.interpolate(
                recons_flat, size=(H, W), mode="bilinear", align_corners=False
            )

        # --- 클램프 (0~1) ---
        videos_flat = videos_flat.clamp(0, 1)
        recons_flat = recons_flat.clamp(0, 1)

        # --- Grid 생성 ---
        grid_real = vutils.make_grid(videos_flat, nrow=videos.shape[1])
        grid_recon = vutils.make_grid(recons_flat, nrow=recons.shape[1])

        # --- WandB로 업로드 ---
        logger.experiment.log({
            f"train/{title}": [
                wandb.Image(grid_real, mode='RGB', caption=f"Original (step {step})"),
                wandb.Image(grid_recon, mode='RGB', caption=f"Reconstructed (step {step})")
            ]
        })
        print(f"[viz-grid] logged at step {step}")

    except Exception as e:
        print(f"[viz-grid] skipped due to error: {e}")


@torch.no_grad()
def log_video_recon_gif(videos, recons, logger, step, max_n=2, fps=4, title="video_recon"):
    """
    Log reconstructed video pairs (original vs recon) as side-by-side GIFs.
    Automatically resizes recon frames to match originals.
    """
    try:
        if isinstance(videos, np.ndarray):
            videos = torch.from_numpy(videos)
        if isinstance(recons, np.ndarray):
            recons = torch.from_numpy(recons)

        videos = videos.detach().cpu().clamp(0, 1)
        recons = recons.detach().cpu().clamp(0, 1)

        B = min(videos.size(0), max_n)
        videos = videos[:B]
        recons = recons[:B]

        # Ensure consistent dim ordering
        if videos.shape[2] not in [1, 3]:
            videos = videos.permute(0, 2, 1, 3, 4)
        if recons.shape[2] not in [1, 3]:
            recons = recons.permute(0, 2, 1, 3, 4)

        for i in range(B):
            frames = []
            for t in range(videos.size(1)):
                frame_r = (videos[i, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frame_g = (recons[i, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # ✅ 해상도 일치
                if frame_r.shape != frame_g.shape:
                    frame_g = cv2.resize(frame_g, (frame_r.shape[1], frame_r.shape[0]), interpolation=cv2.INTER_LINEAR)

                combined = np.concatenate([frame_r, frame_g], axis=1)
                frames.append(combined)

            mp4_path = f"./tmp/{title}_{i}_step{step}.mp4"
            imageio.mimsave(mp4_path, frames, fps=fps, codec='libx264')
            wandb.log({
                f"train/{title}_{i}": wandb.Video(mp4_path, caption=f"Recon #{i}", fps=fps)
            })
        print(f"[viz-gif] logged {B} samples at step {step}")

    except Exception as e:
        print(f"[viz-gif] skipped due to error: {e}")

def match_target_to_recon(video_target, video_recon):
    """
    video_target: [B, T, C, Ht, Wt]
    video_recon : [B, T, C, Hr, Wr]  (decoder 출력 permute 후)
    -> target을 recon 해상도(Hr,Wr)로 다운샘플해서 반환
    """
    B, T, C, Hr, Wr = video_recon.shape
    _, Tt, Ct, Ht, Wt = video_target.shape
    assert C == Ct, f"channel mismatch: recon C={C}, target C={Ct}"
    assert T == Tt, f"T mismatch: recon T={T}, target T={Tt}"

    video_target_ds = F.interpolate(
        video_target.reshape(-1, C, Ht, Wt),  # [B*T, C, Ht, Wt]
        size=(Hr, Wr),
        mode="bilinear",
        align_corners=False
    ).reshape(B, T, C, Hr, Wr).to(video_recon.dtype)
    return video_target_ds

# [B, T, C, H, W] -> 시각화용 [B, T, 3, H, W]
def to_vis_motion(diff_btchw: torch.Tensor) -> torch.Tensor:
    # 1) 채널 평균 + 절댓값
    x = diff_btchw.mean(dim=2).abs()                 # [B, T, H, W]
    # 2) per-sample min-max (각 B마다)
    B, T, H, W = x.shape
    x = x.view(B, -1)
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    x = ((x - x_min) / (x_max - x_min + 1e-6)).view(B, T, H, W)
    # 3) 1채널 → 3채널 복제
    x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)         # [B, T, 3, H, W]
    return x.clamp(0, 1)
