import cv2
import torch
import numpy as np
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel

# ================================
# 1) PRIMUS 로드
# ================================
import sys, os
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

from transformers.models.clip.modeling_clip import CLIPAttention
CLIPAttention._attn_implementation = "eager"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline_modules.primus import PRIMUSLightningModule

import torch

# CLIP (ViT-B/32) 정규화 값
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

def denormalize_clip(tensor):
    """
    CLIP 정규화된 텐서를 0~1 범위의 이미지 텐서 ([C, H, W] 또는 [T, C, H, W])로 변환합니다.
    """
    # 텐서의 마지막 세 차원이 [C, H, W]라고 가정합니다.
    C_dim = tensor.shape[-3] 
    
    mean = CLIP_MEAN.to(tensor.device).view(C_dim, 1, 1)
    std = CLIP_STD.to(tensor.device).view(C_dim, 1, 1)
    
    # 텐서의 모든 차원에 대해 mean과 std를 확장하여 적용합니다.
    # 예: tensor가 [T, C, H, W]면, mean/std가 [T, C, H, W]로 확장되어 적용됩니다.
    
    # 역공식: x_raw = x_norm * std + mean
    # PyTorch의 브로드캐스팅 덕분에 shape을 쉽게 맞출 수 있습니다.
    denorm_tensor = tensor * std + mean
    
    return torch.clamp(denorm_tensor, 0.0, 1.0)

ckpt_path = "/home/jaemo/Method/checkpoints/primus/HWU-USP/last-v7.ckpt"
device = "cuda:0"

ckpt = torch.load(ckpt_path, map_location="cpu")

# video_model.* 제거
keys_to_remove = [k for k in ckpt["state_dict"].keys() if k.startswith("video_model")]
for k in keys_to_remove:
    del ckpt["state_dict"][k]

model = PRIMUSLightningModule.load_from_checkpoint(
    ckpt_path,
    state_dict=ckpt["state_dict"],
)
model = model.to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


# ================================
# 2) CLIP Preprocessor 정의
# ================================
clip_process = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


# ================================
# 3) Video 로드 (CLIP 전용)
# ================================
def load_video_tensor_for_clip(path, target_frames=20):
    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    total = len(frames)
    print(f"[DEBUG] Raw frames: {total}")

    if total < target_frames:
        idxs = np.linspace(0, total - 1, target_frames).astype(int)
    else:
        step = total / target_frames
        idxs = (np.arange(target_frames) * step).astype(int)
    

    frames = [frames[i] for i in idxs]
    print(f"[DEBUG] Step-sampled frames: {len(frames)}")

    processed_list = []
    for f in frames:
        pv = clip_process(images=f, return_tensors="pt")["pixel_values"]
        processed_list.append(pv)

    video_tensor = torch.cat(processed_list, dim=0)  # [T,3,H,W]
    return video_tensor.unsqueeze(0)                 # [1,T,3,H,W]


# ================================
# 4) CLIP Attention Rollout 개선 버전
# ================================
def clip_attention_rollout(video_btchw):
    """
    CLIP 마지막 layer attention만 사용 (더 선명한 heatmap)
    video_btchw: [B,T,3,224,224]
    """
    B, T, C, H, W = video_btchw.shape

    # flatten batch
    video_flat = video_btchw.reshape(B*T, 3, H, W)

    # CLIP backbone
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        output_attentions=True
    ).to(device)
    clip_model.eval()

    outputs = clip_model(
        video_flat,
        output_attentions=True,
        return_dict=True
    )

    attns = outputs.attentions         # list of 12 layers
    last_attn = attns[-1]              # [BT, heads, 50, 50]
    attn_mean = last_attn.mean(1)      # [BT, 50, 50]

    # CLS → PATCH relevance
    cls_to_patch = attn_mean[:, 0, 1:]   # [BT, 49]
    S = int(49 ** 0.5)                   # 7x7 grid
    cls_to_patch = cls_to_patch.reshape(B*T, S, S)

    overlays = []
    import matplotlib.cm as cm

    for idx in range(B * T):
        t = idx % T

        heat = cls_to_patch[idx].detach().cpu().numpy()
        # contrast enhancement
        if heat.max() - heat.min() < 1e-6:
            heat[:] = 0.0
        else:
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            heat = heat ** 0.7       # 강조(감마)

        heat_rgb = cm.jet(heat)[..., :3]

        heat_t = torch.from_numpy(heat_rgb).permute(2,0,1)[None].float()
        heat_up = F.interpolate(heat_t, (H,W), mode="bilinear")[0].permute(1,2,0).numpy()

        frame = video_btchw[0, t].detach().cpu().permute(1,2,0).numpy()

        overlay = 0.7 * heat_up + 0.3 * frame
        overlay_uint8 = (np.clip(overlay, 0,1) * 255).astype(np.uint8)

        overlays.append(overlay_uint8)

    return overlays


# ================================
# 5) 실행
# ================================
if __name__ == "__main__":

    video_path = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/trim_2s_video_cropped/S1-Drill/S1-Drill_002247000_002249000.mp4"
    sample_id = os.path.basename(video_path).replace(".mp4", "")

    video_tensor = load_video_tensor_for_clip(video_path).to(device)

    #
   # ⭐ 첫 번째 배치 (0), 첫 번째 시간 (0) 프레임 추출
    # ==========================================================
    
    # video_tensor[0, 0]는 [C, H, W] 형태입니다.
    first_frame_tensor = video_tensor[0, 10].detach().cpu() 

    # 1. CLIP 비정규화 적용 (0~1 범위로 복원)
    # denormalize_clip 함수는 [C, H, W]를 받아 [C, H, W]로 반환합니다.
    clean_frame_0_1 = denormalize_clip(first_frame_tensor) 

    # 2. [C, H, W] -> [H, W, C] (NumPy 호환)
    clean_frame_numpy = clean_frame_0_1.permute(1, 2, 0).numpy() # float32, 0~1
    
    # 3. [0, 1] -> [0, 255] (uint8) 변환
    frame_to_save = (clean_frame_numpy * 255).astype(np.uint8)

    # 4. 저장
    save_dir = f"./primus/{sample_id}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "first_frame_B0_T0_clean.png")

    # OpenCV는 BGR 순서를 사용하므로 RGB -> BGR 변환 필수
    cv2.imwrite(save_path, cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR))

    print(f"[DEBUG] 첫 번째 프레임 (B=0, T=0) 저장 완료: {save_path}")

    overlays = clip_attention_rollout(video_tensor)

    save_dir = f"./primus/{sample_id}/"
    os.makedirs(save_dir, exist_ok=True)

    for i, img in enumerate(overlays):
        cv2.imwrite(
            os.path.join(save_dir, f"frame_{i:03d}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

    print(f"[Done] Saved {len(overlays)} overlay frames to {save_dir}")
