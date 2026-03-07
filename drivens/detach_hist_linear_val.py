import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from model import SharedEncoder, SceneEncoder, ObjectEncoder, CosineProj  # noqa: E402


DEFAULT_DATA_ROOT = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window"
DEFAULT_DATA_JSON = os.path.join(DEFAULT_DATA_ROOT, "action/linear_val.json")
DEFAULT_CKPT = "/home/jaemo/Method/checkpoints/method/Opportunity++/last.ckpt"
DEFAULT_OUTPUT_DIR = "/home/jaemo/Method/drivens/detach_linear_val_hist"


class LegacyMotionEncoder(nn.Module):
    """Checkpoint-compatible 3D Conv motion encoder."""

    def __init__(self, in_channels=3, base_dim=32, latent_dim=256, use_cosine_proj=True, use_flow=False):
        super().__init__()
        self.use_flow = use_flow
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_dim, base_dim * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_dim * 2, latent_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj = CosineProj(latent_dim, latent_dim) if use_cosine_proj else nn.Linear(latent_dim, latent_dim)

    def forward(self, videos, flows=None):
        video_diff = videos[:, 1:] - videos[:, :-1]  # [B, T-1, 3, H, W]
        if self.use_flow:
            if not isinstance(flows, torch.Tensor):
                raise ValueError("flows must be tensor when use_flow=True")
            if flows.shape[1] > video_diff.shape[1]:
                flows = flows[:, : video_diff.shape[1]]
            elif flows.shape[1] < video_diff.shape[1]:
                flows = torch.cat([flows, flows[:, -1:, :, :, :]], dim=1)
            x = torch.cat([video_diff, flows], dim=2)  # [B, T-1, 5, H, W]
        else:
            x = video_diff

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T-1, H, W]
        feature_5d = self.backbone(x)
        feat = self.spatial_pool(feature_5d).squeeze(-1).squeeze(-1)  # [B, D, T']
        v_motion = feat.mean(dim=2)  # [B, D]
        v_motion = self.proj(v_motion)
        return v_motion, feature_5d


class LegacyVisionModel(nn.Module):
    """Checkpoint-compatible VisionModel (appearance + legacy motion)."""

    def __init__(self, in_channels=3, base_dim=64, latent_dim=256, use_flow=False):
        super().__init__()
        self.shared_encoder = SharedEncoder(in_channels=in_channels, base_dim=base_dim, out_dim=latent_dim)
        self.scene_encoder = SceneEncoder(in_dim=latent_dim, latent_dim=latent_dim)
        self.object_encoder = ObjectEncoder(in_dim=latent_dim, latent_dim=latent_dim)
        self.proj = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = nn.LayerNorm(latent_dim * 2)
        self.fuse = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim))

        motion_in_channels = in_channels + 2 if use_flow else in_channels
        self.motion_encoder = LegacyMotionEncoder(
            in_channels=motion_in_channels,
            base_dim=base_dim // 2,
            latent_dim=latent_dim,
            use_cosine_proj=True,
            use_flow=use_flow,
        )

    def forward(self, video, flows):
        bsz, t_in, c, h, w = video.shape
        video_flat = video.view(bsz * t_in, c, h, w)
        shared_feat = self.shared_encoder(video_flat)
        _, d_dim, hf, wf = shared_feat.shape

        avg_shared_feat_2d = shared_feat.view(bsz, t_in, d_dim, hf, wf).mean(dim=1)
        shared_feat = avg_shared_feat_2d

        v_scene = self.scene_encoder(shared_feat)
        v_object = self.object_encoder(shared_feat)
        fused = torch.cat([v_scene, v_object], dim=1)
        v_appearance = self.fuse(fused)

        v_motion, feature_map_3d = self.motion_encoder(videos=video, flows=flows)
        v_app_norm = F.normalize(v_appearance, dim=1)
        v_mot_norm = F.normalize(v_motion, dim=1)
        with torch.no_grad():
            v_app_norm = v_app_norm.detach()
        z_video_online = torch.cat([v_app_norm, v_mot_norm], dim=1)

        return {
            "v_appearance": v_appearance,
            "v_motion": v_motion,
            "z_video_online": z_video_online,
            "motion_feature_map": feature_map_3d,
            "spatial_feature_map": avg_shared_feat_2d,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Detach negative-pair cosine histogram on linear_val."
    )
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-json", type=str, default=DEFAULT_DATA_JSON)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="v_appearance",
        choices=["z_video_online", "v_appearance", "v_motion"],
        help="Which detach embedding to use for pairwise cosine histogram.",
    )
    parser.add_argument(
        "--clip-range-01",
        action="store_true",
        help="Clip cosine similarities to [0, 1] before histogram.",
    )
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_video_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    embedding_dim = int(hparams.get("embedding_dim", 256))
    use_flow = bool(hparams.get("use_flow", False))

    model = LegacyVisionModel(latent_dim=embedding_dim, use_flow=use_flow).to(device)
    model.eval()

    raw_state = ckpt.get("state_dict", ckpt)
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint format at {ckpt_path}: {type(raw_state)}")

    video_state = {}
    for key, value in raw_state.items():
        if key.startswith("video_model."):
            video_state[key[len("video_model."):]] = value

    if not video_state:
        raise RuntimeError(f"No video_model.* keys found in checkpoint: {ckpt_path}")

    model_state = model.state_dict()
    skipped_shape = []
    for key, value in video_state.items():
        if key in model_state and model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
    if skipped_shape:
        raise RuntimeError(f"Unexpected shape mismatch in legacy loader: {skipped_shape[:5]}")

    missing, unexpected = model.load_state_dict(video_state, strict=False)
    print(f"[INFO] checkpoint loaded: {ckpt_path}")
    print(f"[INFO] loaded video keys: {len(video_state)}")
    print(f"[INFO] missing video keys after load: {len(missing)}")
    print(f"[INFO] unexpected video keys after load: {len(unexpected)}")
    return model


def extract_v_appearance_only(model, video):
    # Mirrors VisionModel.forward appearance branch but skips motion branch.
    bsz, t_in, c, h, w = video.shape
    video_flat = video.view(bsz * t_in, c, h, w)
    shared_feat = model.shared_encoder(video_flat)  # [B*T, D, Hf, Wf]
    _, d_dim, hf, wf = shared_feat.shape
    shared_feat = shared_feat.view(bsz, t_in, d_dim, hf, wf).mean(dim=1)
    v_scene = model.scene_encoder(shared_feat)
    v_object = model.object_encoder(shared_feat)
    fused = torch.cat([v_scene, v_object], dim=1)
    return model.fuse(fused)


def load_items_and_labels(data_json):
    with open(data_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("data", [])
    if not items:
        raise ValueError(f"No samples found in {data_json}")
    labels = np.array([int(item["label"]) for item in items], dtype=np.int64)
    return items, labels


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


def collect_clip_embeddings(model, items, labels, data_root, num_frames, img_size, embedding_key, max_samples, device):
    n_total = len(items)
    n_use = n_total if max_samples <= 0 else min(max_samples, n_total)

    embeddings = []
    used_labels = []
    for idx in range(n_use):
        item = items[idx]
        rel_path = item["frame_path"]
        video_path = rel_path if os.path.isabs(rel_path) else os.path.join(data_root, rel_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")

        video = load_video_tensor(video_path, img_size=img_size, target_frames=num_frames).to(device)
        bsz, t_in, _, h, w = video.shape
        flows = torch.zeros((bsz, t_in, 2, h, w), dtype=video.dtype, device=device)

        with torch.no_grad():
            if embedding_key == "v_appearance":
                emb_t = extract_v_appearance_only(model, video)[0]
            else:
                out = model(video, flows)
                if embedding_key not in out:
                    raise KeyError(f"Embedding key '{embedding_key}' not found. keys={list(out.keys())}")
                emb_t = out[embedding_key][0]
            emb = emb_t.detach().cpu().numpy().astype(np.float32)

        embeddings.append(emb)
        used_labels.append(int(labels[idx]))

    return np.stack(embeddings, axis=0), np.array(used_labels, dtype=np.int64)


def compute_negative_pair_cosine(embeddings, labels):
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = []
    n = emb.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            sims.append(float(np.dot(emb[i], emb[j])))
    if not sims:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(sims, dtype=np.float32)


def save_histogram(sims, out_png, bins, clip_range_01):
    if clip_range_01:
        sims_plot = np.clip(sims, 0.0, 1.0)
        xlim = (0.0, 1.0)
    else:
        sims_plot = sims
        xlim = (-1.0, 1.0)

    plt.figure(figsize=(7.2, 5.0))
    plt.hist(
        sims_plot,
        bins=bins,
        range=xlim,
        density=True,
        color="#1f77b4",
        alpha=0.65,
        edgecolor="black",
        linewidth=0.35,
        label="Detach",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency (Density)")
    plt.title("Detach Negative-Pair Cosine Similarity (linear_val)")
    plt.xlim(*xlim)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)

    print(f"[INFO] device: {device}")
    print(f"[INFO] seed: {args.seed}")
    print(f"[INFO] embedding_key: {args.embedding_key}")

    model = load_video_model_from_ckpt(args.checkpoint_path, device=device)

    items, labels = load_items_and_labels(args.data_json)
    embeddings, used_labels = collect_clip_embeddings(
        model=model,
        items=items,
        labels=labels,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        embedding_key=args.embedding_key,
        max_samples=args.max_samples,
        device=device,
    )
    sims = compute_negative_pair_cosine(embeddings, used_labels)
    if sims.size == 0:
        raise RuntimeError("No negative pairs found. Try increasing --max-samples.")

    out_png = os.path.join(args.output_dir, "detach_linear_val_negative_pairs_hist.png")
    out_npy = os.path.join(args.output_dir, "detach_linear_val_negative_pairs_cosine.npy")
    out_json = os.path.join(args.output_dir, "detach_linear_val_negative_pairs_stats.json")

    save_histogram(sims=sims, out_png=out_png, bins=args.bins, clip_range_01=args.clip_range_01)
    np.save(out_npy, sims)

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(used_labels, return_counts=True))}
    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_json": args.data_json,
        "data_root": args.data_root,
        "checkpoint_path": args.checkpoint_path,
        "seed": args.seed,
        "device": str(device),
        "embedding_key": args.embedding_key,
        "num_total_dataset_samples": int(len(items)),
        "num_used_samples": int(len(used_labels)),
        "label_counts": label_counts,
        "num_negative_pairs": int(len(sims)),
        "cosine_mean": float(np.mean(sims)),
        "cosine_std": float(np.std(sims)),
        "cosine_min": float(np.min(sims)),
        "cosine_max": float(np.max(sims)),
        "outputs": {
            "hist_png": out_png,
            "cosine_npy": out_npy,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("[DONE] histogram saved")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
