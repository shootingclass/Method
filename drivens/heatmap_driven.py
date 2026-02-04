import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose
from transformers import X3DModel
from matplotlib import cm

# ===========================================
# 1) 비디오 로더 (20프레임 sampling)
# ===========================================
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

    # sampling
    idxs = np.linspace(0, len(frames)-1, target_frames).astype(int)
    frames = [frames[i] for i in idxs]

    transform = Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    video = torch.stack([transform(f) for f in frames], dim=1)  # [3,T,H,W]
    return video.unsqueeze(0)  # [1,3,T,H,W]


# ===========================================
# 2) Grad-CAM for X3D
# ===========================================
class X3D_CAM:
    def __init__(self, device="cuda"):
        self.device = device
        from pytorchvideo.models.hub import x3d_s

        self.model = x3d_s(pretrained=True)

        # hook layers
        self.gradients = None
        self.activations = None

        target_layer = self.model.blocks[-1].res_blocks[-1].branch2.c
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, video):
        video = video.to(self.device)  # [1,3,T,H,W]

        out = self.model(video)
        logits = out.last_hidden_state.mean()   # dummy scalar
        logits.backward()

        grads = self.gradients           # [1,C,t,h,w]
        acts = self.activations          # [1,C,t,h,w]

        weights = grads.mean(dim=[2,3,4], keepdim=True)   # [1,C,1,1,1]
        cam = (acts * weights).sum(1, keepdim=False)      # [1,t,h,w]

        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-6)
        return cam[0].cpu().numpy()      # [T,H,W]


# ===========================================
# 3) Heatmap overlay
# ===========================================
def overlay_heatmap(frame, heat):
    heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
    heat_rgb = cm.jet(heat)[..., :3]

    overlay = 0.45*heat_rgb + 0.55*(frame / 255.0)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8


# ===========================================
# 4) 실행 함수
# ===========================================
def run(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_id = os.path.basename(video_path).replace(".mp4","")
    save_dir = f"./heatmap_x3d/{sample_id}/"
    os.makedirs(save_dir, exist_ok=True)

    print("[INFO] Loading video...")
    video = load_video_tensor(video_path).to(device)   # [1,3,T,H,W]

    # raw frames for rendering
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        raw_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    # 20 프레임 sampling (raw frame도 동일하게)
    idxs = np.linspace(0, len(raw_frames)-1, video.shape[2]).astype(int)
    raw_frames = [raw_frames[i] for i in idxs]

    print("[INFO] Running X3D Grad-CAM...")
    cam_extractor = X3D_CAM(device)
    heatmaps = cam_extractor(video)   # [T,H,W]

    for i, heat in enumerate(heatmaps):
        frame = raw_frames[i]
        overlay = overlay_heatmap(frame, heat)
        cv2.imwrite(os.path.join(save_dir, f"frame_{i:03d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"[Done] Saved {len(heatmaps)} heatmap frames → {save_dir}")


# ===========================================
# 5) 진입점
# ===========================================
if __name__ == "__main__":
    video_path = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/trim_2s_video_cropped/sandwich/sandwich_s03_0000031000_0000033000.mp4"
    run(video_path)
