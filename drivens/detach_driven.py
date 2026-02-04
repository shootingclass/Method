import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

# ----------------------------
# 1) Method LightningModule 로드
# ----------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from method import MethodLightningModule
import matplotlib.cm as cm

device = "cuda:0"

ckpt_path = "/home/jaemo/Method/checkpoints/method/HWU-USP/method.ckpt"
print("Loading detach model ...")

model = MethodLightningModule.load_from_checkpoint(ckpt_path, strict=False)
model = model.to(device)
model.eval()
print("Model loaded.")

# --------------------------------------
# 2) Video loading (20 frames sampling)
# --------------------------------------
def load_video_tensor(path, img_size=224, target_frames=20):
    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Force to 20 frames
    idxs = np.linspace(0, len(frames)-1, target_frames).astype(int)
    frames = [frames[i] for i in idxs]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ])

    tensor_frames = torch.stack([transform(f) for f in frames], dim=0)  # [20,3,H,W]
    return tensor_frames.unsqueeze(0)  # [1,20,3,H,W]


# --------------------------------------
# 3) Heatmap utilities
# --------------------------------------
def to_colormap_norm(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    rgb = cm.jet(arr)[..., :3]
    return rgb

# --------------------------------------
# 4) Run detach visualizer
# --------------------------------------
def run_detach_visualize(video_path):
    video = load_video_tensor(video_path).to(device)   # [1,20,3,H,W]
    B, T, _, H, W = video.shape

    # flows는 사용하지 않으므로 0으로 생성
    flows = torch.zeros((B, T, 2, H, W), device=device)

    with torch.no_grad():
        out = model.video_model(video, flows)

    motion_map = out["motion_feature_map"][0]     # [C, T', H', W']
    spatial_map = out["spatial_feature_map"][0]   # [C, H', W']

    C_m, T_m, Hm, Wm = motion_map.shape

    # --- Precompute appearance heatmap (same for all frames)
    spatial_mean = spatial_map.mean(dim=0).cpu().numpy()
    spatial_rgb = to_colormap_norm(spatial_mean)
    spatial_rgb_t = torch.from_numpy(spatial_rgb).permute(2,0,1)[None].float()

    spatial_up = F.interpolate(
        spatial_rgb_t, size=(H,W), mode='bilinear', align_corners=False
    )[0].permute(1,2,0).cpu().numpy()

    # -------------------------------
    # Directory setup
    # -------------------------------
    sample_id = os.path.basename(video_path).replace(".mp4", "")
    save_root = f"./detach/{sample_id}/"
    motion_dir = os.path.join(save_root, "motion")
    spatial_dir = os.path.join(save_root, "spatial")
    os.makedirs(motion_dir, exist_ok=True)
    os.makedirs(spatial_dir, exist_ok=True)

    # -------------------------------
    # Save overlays separately
    # -------------------------------
    T_use = min(T, T_m)

    for t in range(T_use):

        # Frame
        frame = video[0, t].detach().cpu().permute(1,2,0).numpy()
        frame = np.clip(frame, 0, 1)

        # -------------------------------------------------
        # 1) Motion overlay
        # -------------------------------------------------
        mot_t = motion_map[:, t].mean(dim=0).cpu().numpy()
        mot_rgb = to_colormap_norm(mot_t)
        mot_rgb_t = torch.from_numpy(mot_rgb).permute(2,0,1)[None].float()

        mot_up = F.interpolate(
            mot_rgb_t, size=(H,W), mode="bilinear", align_corners=False
        )[0].permute(1,2,0).cpu().numpy()

        motion_overlay = (0.6 * frame + 0.4 * mot_up)
        motion_overlay = np.clip(motion_overlay, 0, 1)
        motion_uint8 = (motion_overlay * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(motion_dir, f"frame_{t:03d}.png"),
            cv2.cvtColor(motion_uint8, cv2.COLOR_RGB2BGR)
        )

        # -------------------------------------------------
        # 2) Spatial overlay (appearance)
        # -------------------------------------------------
        spatial_overlay = (0.6 * frame + 0.4 * spatial_up)
        spatial_overlay = np.clip(spatial_overlay, 0, 1)
        spatial_uint8 = (spatial_overlay * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(spatial_dir, f"frame_{t:03d}.png"),
            cv2.cvtColor(spatial_uint8, cv2.COLOR_RGB2BGR)
        )

    print(f"[Done] Saved {T_use} motion frames → {motion_dir}")
    print(f"[Done] Saved {T_use} spatial frames → {spatial_dir}")


# --------------------------------------
# 5) Run
# --------------------------------------
video_path = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/trim_2s_video_cropped/tea/tea_s12_0000026000_0000028000.mp4"
run_detach_visualize(video_path)
