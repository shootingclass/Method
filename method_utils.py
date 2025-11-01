import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

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
<<<<<<< Updated upstream
import matplotlib.pyplot as plt
=======
import numpy as np
import cv2

def overlay_motion_heatmap(video, motion_residual, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    원본 영상 위에 motion intensity heatmap을 덧씌움.

    Args:
        video: [B, T, 3, H, W] (0~1 float or 0~255 uint8)
        motion_residual: [B, T, 3, H, W] (video - app_recon 등)
        alpha: heatmap overlay 비율
        colormap: OpenCV 컬러맵 (e.g., cv2.COLORMAP_JET)
    Returns:
        overlay_videos: list of np.ndarray [T, H, W, 3] (uint8)
    """

    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    if isinstance(motion_residual, torch.Tensor):
        motion_residual = motion_residual.detach().cpu().numpy()

    B, T, C, H, W = video.shape
    overlay_results = []

    for b in range(B):
        vid_frames = []
        for t in range(T):
            frame = video[b, t].transpose(1, 2, 0)  # [H, W, 3]
            motion = motion_residual[b, t].transpose(1, 2, 0)

            # --- motion intensity map ---
            mag = np.mean(np.abs(motion), axis=2)  # [H, W]
            mag = (mag - mag.min()) / (mag.max() + 1e-6)
            mag = (mag * 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(mag, colormap)
            frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)

            overlay = cv2.addWeighted(frame_uint8, 1 - alpha, heatmap, alpha, 0)
            vid_frames.append(overlay)
        overlay_results.append(np.stack(vid_frames))

    return overlay_results

import torch
import torchvision.utils as vutils
import numpy as np
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
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
=======
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

class CovarianceAlignmentLoss(nn.Module):
    """
    DiCoSA 논문의 Intra-Concept Alignment Loss (L_A) 구현체.
    [참고: Equation (4), (5), (7)]
    
    두 피처(positive pair) 간의 정규화된 공분산(covariance)을 
    1에 가깝게 만들어 Mutual Information을 최대화합니다.
    """
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def _batch_normalize(self, x):
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False) 
        std = (var + self.epsilon).sqrt()
        z = (x - mean) / std
        return z

    def forward(self, feature_a, feature_b):
        z_a = self._batch_normalize(feature_a)
        z_s = self._batch_normalize(feature_b)
        
        # C_positive = E[(z_a)^T * z_s]
        C_positive = (z_a * z_s).sum(dim=1).mean()
        
        # loss = (1 - C_positive)^2
        loss = (1.0 - C_positive).pow(2)
        return loss
    

# ---------------------------------------------------------
# Temporal pooling utilities for viz/alignment after train
# ---------------------------------------------------------
class TemporalAttentionPool(nn.Module):
    """Lightweight attention over time: returns [B, D] and weights [B, T', 1]."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):  # x: [B, T', D]
        w = self.score(x)                 # [B, T', 1]
        a = torch.softmax(w, dim=1)       # [B, T', 1]
        pooled = (x * a).sum(dim=1)       # [B, D]
        return pooled, a                  # for visualization

def get_motion_repr(v_motion_seq, mode="mean", attn_pool: TemporalAttentionPool = None):
    """
    v_motion_seq: [B, T', D]
    returns: [B, D] (global motion vector), optional weights
    """
    if mode == "mean":
        return v_motion_seq.mean(dim=1), None
    elif mode == "max":
        return v_motion_seq.max(dim=1).values, None
    elif mode == "attn":
        assert attn_pool is not None, "Provide attn_pool for mode='attn'"
        pooled, weights = attn_pool(v_motion_seq)
        return pooled, weights
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ------------------------------------------------------------
# Helper utils
# ------------------------------------------------------------
def downsample_video(video, size=(112, 112)):
    B, T, C, H, W = video.shape
    x = video.permute(0, 2, 1, 3, 4)
    x = F.interpolate(x, size=(T, *size), mode="trilinear", align_corners=False)
    return x.permute(0, 2, 1, 3, 4)

def repeat_to_T(x, T):
    return x.expand(-1, T, -1)

import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import wandb

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

def log_optical_flow_overlay_to_wandb(video, flows, wandb_key="optical_flow_overlay", stride=10, scale=5, fps=10):
    """
    Optical flow를 비디오 프레임 위에 overlay하여 WandB에 mp4로 업로드.

    Args:
        video (Tensor): [B, T, 3, H, W]
        flows (Tensor): [B, T, 2, H, W]
    """
    assert video.dim() == 5 and flows.dim() == 5, "video, flows는 [B, T, C, H, W] 형태여야 합니다."
    B, T, _, H, W = video.shape
    assert B == 1, "현재는 batch=1만 지원합니다."

    frames = []

    for t in range(T):
        frame = video[0, t].permute(1, 2, 0).detach().cpu().numpy()
        flow = flows[0, min(t, flows.shape[1] - 1)].detach().cpu().numpy()

        frame_disp = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
        frame_disp = (frame_disp * 255).astype(np.uint8)

        X, Y = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
        u = flow[0, ::stride, ::stride]
        v = flow[1, ::stride, ::stride]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame_disp)
        ax.quiver(X, Y, u, -v, color='r', angles='xy', scale_units='xy', scale=scale, width=0.002)
        ax.axis('off')
        plt.tight_layout(pad=0)

        # ✅ 여기 수정됨 — 최신 matplotlib에서 작동
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA → RGB
        frames.append(img)
        plt.close(fig)

    # numpy → wandb.Video
    video_tensor = np.stack(frames)
    video_tensor = np.transpose(video_tensor, (0, 3, 1, 2))  # [T, C, H, W]
    wandb_video = wandb.Video(video_tensor, fps=fps, format="mp4")

    wandb.log({wandb_key: wandb_video})
    print(f"✅ Optical flow overlay video logged to WandB ({wandb_key})")
>>>>>>> Stashed changes
