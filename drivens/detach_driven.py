import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:
    from decord import VideoReader
except ModuleNotFoundError:  # pragma: no cover
    VideoReader = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from method import MethodLightningModule  # noqa: E402


DEFAULT_DATA_ROOT = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window"
DEFAULT_DATA_JSON = os.path.join(DEFAULT_DATA_ROOT, "action/linear_val.json")
DEFAULT_CKPT = "/home/jaemo/Method/checkpoints/method/Opportunity++/last.ckpt"
DEFAULT_OUTPUT_DIR = "/home/jaemo/Method/drivens/detach_linear_val"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_sample(args):
    if args.video_path:
        abs_video = args.video_path
        if not os.path.isabs(abs_video):
            abs_video = os.path.join(args.data_root, abs_video)
        return {
            "sample_index": None,
            "video_id": Path(abs_video).stem,
            "frame_path": args.video_path,
            "label": None,
            "video_path": abs_video,
        }

    data = load_json(args.data_json)
    items = data.get("data", [])
    if not items:
        raise ValueError(f"No 'data' samples found in: {args.data_json}")
    if args.sample_index < 0 or args.sample_index >= len(items):
        raise IndexError(
            f"sample_index out of range: {args.sample_index} (dataset size={len(items)})"
        )

    sample = items[args.sample_index]
    rel_frame_path = sample["frame_path"]
    video_path = rel_frame_path
    if not os.path.isabs(video_path):
        video_path = os.path.join(args.data_root, rel_frame_path)

    return {
        "sample_index": int(args.sample_index),
        "video_id": sample.get("video_id", Path(video_path).stem),
        "frame_path": rel_frame_path,
        "label": sample.get("label"),
        "video_path": video_path,
    }


def load_video_tensor(path, img_size=224, target_frames=20):
    frames = []

    if cv2 is not None:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
    elif VideoReader is not None:
        vr = VideoReader(path)
        frames = [vr[i].asnumpy() for i in range(len(vr))]
    else:
        try:
            from torchvision.io import read_video
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "No video backend available. Install opencv-python or decord."
            ) from exc
        frame_tensor, _, _ = read_video(path, pts_unit="sec")
        frames = frame_tensor.numpy()

    if not frames:
        raise ValueError(f"No frames loaded from video: {path}")

    idxs = np.linspace(0, len(frames) - 1, target_frames).astype(int)
    sampled = [frames[i] for i in idxs]

    tensor_frames = []
    for frame in sampled:
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_t = TF.resize(frame_t, [img_size, img_size], antialias=True)
        tensor_frames.append(frame_t)

    return torch.stack(tensor_frames, dim=0).unsqueeze(0)  # [1, T, 3, H, W]


def to_colormap_norm(arr):
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    x = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # Lightweight jet approximation to avoid matplotlib dependency.
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_first_frame(video, save_path):
    frame = video[0, 0].detach().cpu().permute(1, 2, 0).numpy()
    frame_u8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(frame_u8).save(save_path)


def save_original_frames(video, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for t in range(video.shape[1]):
        frame = video[0, t].detach().cpu().permute(1, 2, 0).numpy()
        frame_u8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(frame_u8).save(os.path.join(save_dir, f"frame_{t:03d}.png"))
    return {"original_frame_dir": save_dir, "num_frames": int(video.shape[1])}


def run_once(args):
    device = choose_device(args.device)
    sample = resolve_sample(args)
    video_path = sample["video_path"]

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    print(f"[INFO] device: {device}")
    print(f"[INFO] sample: {sample['video_id']}")
    print(f"[INFO] loading checkpoint: {args.checkpoint_path}")

    model = MethodLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        strict=False,
        map_location="cpu",
    ).to(device)
    model.eval()

    video = load_video_tensor(
        path=video_path,
        img_size=args.img_size,
        target_frames=args.num_frames,
    ).to(device)
    bsz, t_in, _, h, w = video.shape
    flows = torch.zeros((bsz, t_in, 2, h, w), dtype=video.dtype, device=device)

    with torch.no_grad():
        out = model.video_model(video, flows)

    if "motion_feature_map" not in out or "spatial_feature_map" not in out:
        raise KeyError("Expected keys 'motion_feature_map' and 'spatial_feature_map' not found.")

    motion_map = out["motion_feature_map"][0]   # [C, T', H', W']
    spatial_map = out["spatial_feature_map"][0]  # [C, H', W']

    os.makedirs(args.output_dir, exist_ok=True)
    motion_dir = os.path.join(args.output_dir, "motion_overlay")
    spatial_dir = os.path.join(args.output_dir, "spatial_overlay")
    os.makedirs(motion_dir, exist_ok=True)
    os.makedirs(spatial_dir, exist_ok=True)

    first_frame_path = os.path.join(args.output_dir, "first_frame.png")
    save_first_frame(video, first_frame_path)
    original_info = save_original_frames(video, os.path.join(args.output_dir, "original_frames"))

    spatial_mean = spatial_map.mean(dim=0).detach().cpu().numpy()
    spatial_rgb = to_colormap_norm(spatial_mean)
    spatial_rgb_t = torch.from_numpy(spatial_rgb).permute(2, 0, 1)[None].float()
    spatial_up = F.interpolate(
        spatial_rgb_t, size=(h, w), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0).cpu().numpy()

    t_use = min(t_in, int(motion_map.shape[1]))
    for t in range(t_use):
        frame = video[0, t].detach().cpu().permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0.0, 1.0)

        mot_t = motion_map[:, t].mean(dim=0).detach().cpu().numpy()
        mot_rgb = to_colormap_norm(mot_t)
        mot_rgb_t = torch.from_numpy(mot_rgb).permute(2, 0, 1)[None].float()
        mot_up = F.interpolate(
            mot_rgb_t, size=(h, w), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()

        motion_overlay = np.clip(0.6 * frame + 0.4 * mot_up, 0.0, 1.0)
        spatial_overlay = np.clip(0.6 * frame + 0.4 * spatial_up, 0.0, 1.0)

        motion_u8 = (motion_overlay * 255.0).astype(np.uint8)
        spatial_u8 = (spatial_overlay * 255.0).astype(np.uint8)

        Image.fromarray(motion_u8).save(os.path.join(motion_dir, f"frame_{t:03d}.png"))
        Image.fromarray(spatial_u8).save(os.path.join(spatial_dir, f"frame_{t:03d}.png"))

    result = {
        "sample_index": sample["sample_index"],
        "video_id": sample["video_id"],
        "sample_meta": sample["frame_path"],
        "label": sample["label"],
        "video_path": video_path,
        "checkpoint_path": args.checkpoint_path,
        "video_shape": list(video.shape),
        "flow_shape": list(flows.shape),
        "motion_map_shape": list(motion_map.shape),
        "spatial_map_shape": list(spatial_map.shape),
        "num_saved_frames": int(t_use),
        "first_frame_path": first_frame_path,
        "original_info": original_info,
        "motion_overlay_dir": motion_dir,
        "spatial_overlay_dir": spatial_dir,
    }

    result_path = os.path.join(args.output_dir, "run_once_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"[DONE] saved result: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-run Detach visualizer")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-json", type=str, default=DEFAULT_DATA_JSON)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="Optional direct video path (relative paths are resolved from data_root).",
    )
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run_once(parse_args())
