import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

import mae_driven as md


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MAE negative-pair cosine histogram on linear_val."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/jaemo/Method/checkpoints/evimae/Opportunity++/models/evi_model.299.pth",
    )
    parser.add_argument(
        "--args-json",
        type=str,
        default="/home/jaemo/Method/checkpoints/evimae/Opportunity++/args.json",
    )
    parser.add_argument(
        "--data-json",
        type=str,
        default="/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/action/linear_val.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/jaemo/Method/drivens/mae_linear_val_hist",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--clip-range-01",
        action="store_true",
        help="Clip cosine similarities to [0, 1] before histogram.",
    )
    return parser.parse_args()


def load_labels_from_json(data_json):
    raw = md.load_json(data_json)
    items = raw.get("data", [])
    if not items:
        raise ValueError(f"No data found in json: {data_json}")
    labels = np.array([int(item["label"]) for item in items], dtype=np.int64)
    metas = [item.get("video_id", item.get("frame_path", f"sample_{i}")) for i, item in enumerate(items)]
    return labels, metas


def collect_clip_embeddings(model, dataset, labels, max_samples, device):
    n_total = len(dataset)
    n_use = n_total if max_samples <= 0 else min(max_samples, n_total)

    embeddings = []
    used_labels = []
    used_metas = []
    for idx in range(n_use):
        imu_input, video_input, _third, _label = dataset[idx]
        imu_input = md.ensure_dim(imu_input, 4).to(device)
        video_input = md.ensure_dim(video_input, 5).to(device)

        del imu_input
        with torch.no_grad():
            video_tokens = md._extract_video_tokens_nomask(model=model, video_input=video_input)
            # Clip embedding: mean over all video tokens.
            emb = video_tokens.mean(dim=1)[0].detach().cpu().numpy()  # [D]

        embeddings.append(emb.astype(np.float32))
        used_labels.append(int(labels[idx]))
        used_metas.append(str(idx))

    return np.stack(embeddings, axis=0), np.array(used_labels, dtype=np.int64), used_metas


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
        color="#d62728",
        alpha=0.65,
        edgecolor="black",
        linewidth=0.35,
        label="MAE",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency (Density)")
    plt.title("MAE Negative-Pair Cosine Similarity (linear_val)")
    plt.xlim(*xlim)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    md.set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device: {device}")
    print(f"[INFO] seed: {args.seed}")

    args_dict = md.load_json(args.args_json)
    video_model_dict, imu_model_dict = md.build_model_dicts(args_dict)

    model = md.evimae_models.EVIMAE(
        norm_pix_loss=args_dict["norm_pix_loss"],
        tr_pos=args_dict["tr_pos"],
        video_model_dict=video_model_dict,
        imu_model_dict=imu_model_dict,
    ).to(device)
    model.eval()
    md.load_checkpoint_state_dict(model, args.checkpoint_path)

    imu_conf = md.build_imu_conf(args_dict)
    dataset = md.evimae_dataloader.EVIDataset(
        args.data_json,
        imu_conf=imu_conf,
        label_csv=args_dict.get("label_csv"),
        video_masking_ratio=args_dict["video_masking_ratio"],
        image_as_video=args_dict.get("image_as_video", False),
    )
    md.set_dataset_deterministic(dataset, img_size=args_dict["video_img_size"])

    labels, metas = load_labels_from_json(args.data_json)
    if len(labels) != len(dataset):
        raise RuntimeError(
            f"Label count mismatch: labels={len(labels)} vs dataset={len(dataset)} "
            f"for {args.data_json}"
        )

    embeddings, used_labels, _used_metas = collect_clip_embeddings(
        model=model,
        dataset=dataset,
        labels=labels,
        max_samples=args.max_samples,
        device=device,
    )
    sims = compute_negative_pair_cosine(embeddings, used_labels)
    if sims.size == 0:
        raise RuntimeError("No negative pairs found. Try increasing --max-samples.")

    out_png = os.path.join(args.output_dir, "mae_linear_val_negative_pairs_hist.png")
    out_npy = os.path.join(args.output_dir, "mae_linear_val_negative_pairs_cosine.npy")
    out_json = os.path.join(args.output_dir, "mae_linear_val_negative_pairs_stats.json")

    save_histogram(
        sims=sims,
        out_png=out_png,
        bins=args.bins,
        clip_range_01=args.clip_range_01,
    )
    np.save(out_npy, sims)

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(used_labels, return_counts=True))}
    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_json": args.data_json,
        "checkpoint_path": args.checkpoint_path,
        "args_json": args.args_json,
        "seed": args.seed,
        "device": str(device),
        "num_total_dataset_samples": int(len(dataset)),
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
