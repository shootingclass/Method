import torch

import cv2
import numpy as np
from torchvision import transforms
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline_modules.comodo.model import VideoTeacher     # 너가 쓰는 COMODO teacher 경로로 수정
import matplotlib.cm as cm
import torch.nn.functional as F


# --------------------------------------------------------
# 1) Load video into tensor [1, T, 3, 224, 224]
# --------------------------------------------------------
def load_video_tensor(path, img_size=224):
    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    frame_tensors = torch.stack([transform(f) for f in frames], dim=0)
    frame_tensors = frame_tensors.unsqueeze(0)  # [1, T, 3, H, W]
    return frame_tensors


# --------------------------------------------------------
# 2) COMODO teacher feature overlay extraction
#    (Single function, no hook)
# --------------------------------------------------------
def comodo_feature_overlay(teacher, video_tensor):
    """
    returns:
        embeddings: [1, D]
        overlays: list of numpy HxWx3 uint8 images
    """
    import torch.nn.functional as F

    B, T, C, H, W = video_tensor.shape
    video_tensor = video_tensor.to(teacher.device)

    # Forward with hidden states
    outputs = teacher.model(
        video_tensor,
        output_hidden_states=True,
        return_dict=True
    )

    hidden = outputs.hidden_states[-1]     # [B, N_tokens, D]
    patch_tokens = hidden[:, 1:, :]        # remove CLS

    # For VideoMAE or Timesformer: N_tokens = T * (H'*W')
    num_tokens = patch_tokens.shape[1]
    spatial_dim = int((num_tokens // T) ** 0.5)  # e.g. 14
    tokens_per_frame = spatial_dim * spatial_dim

    # reshape → [B, T, H', W', D]
    patch_tokens = patch_tokens.reshape(B, T, spatial_dim, spatial_dim, -1)

    overlays = []

    # Frame-by-frame overlay
    for t in range(T):
        feat = patch_tokens[0, t].detach().cpu().numpy()  # [H',W',D]
        heat = feat.mean(-1)                     # [H',W']

        # normalize
        hmin, hmax = heat.min(), heat.max()
        heat_norm = (heat - hmin) / (hmax - hmin + 1e-8)

        # color map
        heat_rgb = cm.get_cmap('jet')(heat_norm)[..., :3]

        # upsample to original resolution
        heat_t = torch.from_numpy(heat_rgb).permute(2,0,1)[None].float()
        heat_up = F.interpolate(
            heat_t, size=(H, W), mode='bilinear', align_corners=False
        )[0].permute(1,2,0).numpy()

        # original frame
        frame = video_tensor[0, t].detach().cpu()
        if frame.min() < 0: frame = (frame + 1) / 2
        frame_np = frame.permute(1, 2, 0).numpy()

        # overlay
        alpha = 0.45
        overlay = (1-alpha)*frame_np + alpha*heat_up
        overlay_uint8 = (np.clip(overlay, 0, 1)*255).astype(np.uint8)

        overlays.append(overlay_uint8)

    # embedding (COMODO original)
    video_hidden_state = outputs.last_hidden_state     # [B,T,dim] or [B,dim]
    if teacher.use_mean_pooling:
        video_embeddings = video_hidden_state.mean(dim=1)
    else:
        video_embeddings = video_hidden_state[:, 0]

    video_embeddings = torch.nn.functional.normalize(video_embeddings, dim=1)

    return video_embeddings, overlays


# --------------------------------------------------------
# 3) main inference
# --------------------------------------------------------
def run_comodo_inference(model_name, ckpt_path, video_path, output_root="./comodo"):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    teacher = VideoTeacher(model_name=model_name, device=device)

    # load video
    video_tensor = load_video_tensor(video_path).to(device)
    T = video_tensor.shape[1]
    idx = torch.linspace(0, T-1, steps=20).long()
    video_tensor = video_tensor[:, idx]

    print(f"[INFO] Video loaded: {video_tensor.shape}")

    # extract embedding + overlays
    emb, overlays = comodo_feature_overlay(teacher, video_tensor)

    sample_name = os.path.basename(video_path).replace(".mp4", "")
    save_dir = os.path.join(output_root, sample_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Saving overlays to {save_dir}")

    for i, img in enumerate(overlays):
        save_path = os.path.join(save_dir, f"frame_{i:03d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"[DONE] Saved {len(overlays)} frames.")


# --------------------------------------------------------
# RUN
# --------------------------------------------------------
if __name__ == "__main__":

    model_name = "facebook/timesformer-base-finetuned-k400"         # 또는 Timesformer 이름
    ckpt_path = "/home/jaemo/Method/checkpoints/comodo/HWU-USP/comodo.ckpt"                          # teacher는 HF pretrained만 사용
    video_path = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/trim_2s_video_cropped/S3-ADL2/S3-ADL2_001219000_001221000.mp4"

    run_comodo_inference(model_name, ckpt_path, video_path)
