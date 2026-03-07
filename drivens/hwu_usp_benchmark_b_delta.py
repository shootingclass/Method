import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

import detach_hist_linear_val as dd
import mae_driven as md


DEFAULT_DATA_ROOT = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window"
DEFAULT_DATA_JSON = (
    "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/"
    "motion_2_priority_test=18/pretrain_cropped_with_flow_detail_prudent.json"
)
DEFAULT_OUTPUT_DIR = "/home/jaemo/Method/drivens/hwu_usp_benchmark_b_delta"

DEFAULT_MAE_CKPT = "/home/jaemo/Method/checkpoints/evimae/HWU-USP/models/evi_model.299.pth"
DEFAULT_MAE_ARGS = "/home/jaemo/Method/checkpoints/evimae/HWU-USP/args.json"
DEFAULT_DETACH_CKPT = "/home/jaemo/Method/checkpoints/method/HWU-USP/last.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "HWU-USP Benchmark B (Inversion): "
            "DeltaS(v)=Sim(v,v_hard)-Sim(v,v_easy), "
            "hard=(same object, opposite open/close), "
            "easy=(different object, same open/close)."
        )
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-json", type=str, default=DEFAULT_DATA_JSON)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--model-type",
        type=str,
        default="both",
        choices=["mae", "detach", "both"],
        help="Which embedding backbone to run.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to reduce multiple hard/easy candidates into a single similarity.",
    )

    parser.add_argument("--mae-checkpoint-path", type=str, default=DEFAULT_MAE_CKPT)
    parser.add_argument("--mae-args-json", type=str, default=DEFAULT_MAE_ARGS)
    parser.add_argument("--mae-num-frames", type=int, default=16)
    parser.add_argument(
        "--mae-embedding-mode",
        type=str,
        default="temporal_diff",
        choices=["clip_mean", "temporal_diff"],
        help=(
            "MAE embedding mode. "
            "temporal_diff uses temporal token differences to emphasize motion."
        ),
    )

    parser.add_argument("--detach-checkpoint-path", type=str, default=DEFAULT_DETACH_CKPT)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="v_appearance",
        choices=["z_video_online", "v_appearance", "v_motion"],
        help="Detach embedding type.",
    )
    return parser.parse_args()


def parse_detail_label(detail_label):
    if not isinstance(detail_label, str):
        return None, None, False
    text = detail_label.strip()
    if not text:
        return None, None, False
    parts = text.split(None, 1)
    action = parts[0].lower()
    if action not in {"open", "close"}:
        return None, None, False
    obj = parts[1].strip() if len(parts) > 1 else ""
    if not obj:
        return None, None, False
    return action, obj, True


def load_items(data_json, max_samples):
    with open(data_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("data", [])
    if not items:
        raise ValueError(f"No items found in {data_json}")
    if max_samples > 0:
        items = items[:max_samples]
    return items


def build_anchor_metadata(items):
    motions = []
    objects = []
    valid_openclose = []
    video_ids = []
    for idx, item in enumerate(items):
        action, obj, valid = parse_detail_label(item.get("detail_label"))
        motions.append(action)
        objects.append(obj)
        valid_openclose.append(valid)
        video_ids.append(item.get("video_id", f"sample_{idx}"))
    return (
        np.asarray(motions, dtype=object),
        np.asarray(objects, dtype=object),
        np.asarray(valid_openclose, dtype=bool),
        video_ids,
    )


def choose_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_sim(values, reduction):
    if values.size == 0:
        raise ValueError("reduce_sim called with empty values.")
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "max":
        return float(np.max(values))
    raise ValueError(f"Unsupported reduction: {reduction}")


def compute_delta_similarity(embeddings, motions, objects, valid_openclose, reduction):
    emb = embeddings.astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim_matrix = emb @ emb.T

    deltas = []
    hard_sims = []
    easy_sims = []
    hard_candidate_counts = []
    easy_candidate_counts = []
    anchor_indices = []

    n = emb.shape[0]
    for i in range(n):
        if not valid_openclose[i]:
            continue

        hard_mask = valid_openclose & (objects == objects[i]) & (motions != motions[i])
        easy_mask = valid_openclose & (motions == motions[i]) & (objects != objects[i])

        hard_cnt = int(np.sum(hard_mask))
        easy_cnt = int(np.sum(easy_mask))
        if hard_cnt == 0 or easy_cnt == 0:
            continue

        sim_hard = reduce_sim(sim_matrix[i, hard_mask], reduction=reduction)
        sim_easy = reduce_sim(sim_matrix[i, easy_mask], reduction=reduction)

        deltas.append(sim_hard - sim_easy)
        hard_sims.append(sim_hard)
        easy_sims.append(sim_easy)
        hard_candidate_counts.append(hard_cnt)
        easy_candidate_counts.append(easy_cnt)
        anchor_indices.append(i)

    if not deltas:
        return {
            "deltas": np.zeros((0,), dtype=np.float32),
            "hard_sims": np.zeros((0,), dtype=np.float32),
            "easy_sims": np.zeros((0,), dtype=np.float32),
            "hard_candidate_counts": np.zeros((0,), dtype=np.int32),
            "easy_candidate_counts": np.zeros((0,), dtype=np.int32),
            "anchor_indices": np.zeros((0,), dtype=np.int32),
            "num_valid_openclose": int(np.sum(valid_openclose)),
            "num_anchors_without_pair": int(np.sum(valid_openclose)),
        }

    used_mask = np.zeros((n,), dtype=bool)
    used_mask[np.asarray(anchor_indices, dtype=np.int32)] = True
    num_anchors_without_pair = int(np.sum(valid_openclose & ~used_mask))
    return {
        "deltas": np.asarray(deltas, dtype=np.float32),
        "hard_sims": np.asarray(hard_sims, dtype=np.float32),
        "easy_sims": np.asarray(easy_sims, dtype=np.float32),
        "hard_candidate_counts": np.asarray(hard_candidate_counts, dtype=np.int32),
        "easy_candidate_counts": np.asarray(easy_candidate_counts, dtype=np.int32),
        "anchor_indices": np.asarray(anchor_indices, dtype=np.int32),
        "num_valid_openclose": int(np.sum(valid_openclose)),
        "num_anchors_without_pair": num_anchors_without_pair,
    }


def plot_delta_histogram(deltas, out_png, model_name, bins, color):
    plt.figure(figsize=(7.6, 5.2))
    plt.hist(
        deltas,
        bins=bins,
        range=(-1.0, 1.0),
        density=True,
        color=color,
        alpha=0.62,
        edgecolor="black",
        linewidth=0.35,
        label=model_name,
    )
    plt.axvline(0.0, linestyle="--", color="black", linewidth=1.2, label="x=0")
    plt.xlabel("Delta Similarity: Sim(v, v_hard) - Sim(v, v_easy)")
    plt.ylabel("Density")
    plt.title(f"{model_name} Benchmark B Delta Similarity (HWU-USP)")
    plt.xlim(-1.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_overlay(mae_deltas, detach_deltas, out_png, bins):
    plt.figure(figsize=(7.6, 5.2))
    plt.hist(
        mae_deltas,
        bins=bins,
        range=(-1.0, 1.0),
        density=True,
        color="#d62728",
        alpha=0.50,
        edgecolor="black",
        linewidth=0.30,
        label="MAE",
    )
    plt.hist(
        detach_deltas,
        bins=bins,
        range=(-1.0, 1.0),
        density=True,
        color="#1f77b4",
        alpha=0.50,
        edgecolor="black",
        linewidth=0.30,
        label="Detach",
    )
    plt.axvline(0.0, linestyle="--", color="black", linewidth=1.2, label="x=0")
    plt.xlabel("Delta Similarity: Sim(v, v_hard) - Sim(v, v_easy)")
    plt.ylabel("Density")
    plt.title("Benchmark B Delta Similarity (HWU-USP)")
    plt.xlim(-1.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def extract_mae_embeddings(args, items, device):
    args_dict = md.load_json(args.mae_args_json)
    video_model_dict, imu_model_dict = md.build_model_dicts(args_dict)
    model = md.evimae_models.EVIMAE(
        norm_pix_loss=args_dict["norm_pix_loss"],
        tr_pos=args_dict["tr_pos"],
        video_model_dict=video_model_dict,
        imu_model_dict=imu_model_dict,
    ).to(device)
    model.eval()
    md.load_checkpoint_state_dict(model, args.mae_checkpoint_path)

    img_size = int(args_dict["video_img_size"])
    embeddings = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    for item in items:
        rel_path = item["frame_path"]
        video_path = rel_path if os.path.isabs(rel_path) else os.path.join(args.data_root, rel_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")

        # dd.load_video_tensor returns [1, T, 3, H, W] in [0,1].
        video_btchw = dd.load_video_tensor(
            video_path,
            img_size=img_size,
            target_frames=args.mae_num_frames,
        ).to(device)
        video_input = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
        video_input = (video_input - mean) / std

        with torch.no_grad():
            tokens = md._extract_video_tokens_nomask(model=model, video_input=video_input)[0]  # [N, D]
            if args.mae_embedding_mode == "clip_mean":
                emb_t = tokens.mean(dim=0)
            elif args.mae_embedding_mode == "temporal_diff":
                # Token layout is [tubelets * spatial, D].
                patch_h = img_size // int(model.video_patch_size)
                patch_w = img_size // int(model.video_patch_size)
                spatial = patch_h * patch_w
                if spatial <= 0 or tokens.shape[0] < spatial:
                    raise RuntimeError(
                        f"Unexpected token shape for temporal_diff: tokens={tokens.shape}, "
                        f"patch_h={patch_h}, patch_w={patch_w}"
                    )
                tubelets = tokens.shape[0] // spatial
                token_3d = tokens[: tubelets * spatial].reshape(tubelets, spatial, -1)
                temporal = token_3d.mean(dim=1)  # [T', D]
                if temporal.shape[0] < 2:
                    raise RuntimeError(
                        f"temporal_diff requires at least 2 tubelets, got {temporal.shape[0]}"
                    )
                emb_t = (temporal[1:] - temporal[:-1]).mean(dim=0)
            else:
                raise ValueError(f"Unsupported mae_embedding_mode: {args.mae_embedding_mode}")
            emb = emb_t.detach().cpu().numpy().astype(np.float32)
        embeddings.append(emb)

    return np.stack(embeddings, axis=0)


def extract_detach_embeddings(args, items, device):
    model = dd.load_video_model_from_ckpt(args.detach_checkpoint_path, device=device)

    embeddings = []
    for item in items:
        rel_path = item["frame_path"]
        video_path = rel_path if os.path.isabs(rel_path) else os.path.join(args.data_root, rel_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")

        video = dd.load_video_tensor(video_path, img_size=args.img_size, target_frames=args.num_frames).to(device)
        bsz, t_in, _, h, w = video.shape
        flows = torch.zeros((bsz, t_in, 2, h, w), dtype=video.dtype, device=device)

        with torch.no_grad():
            if args.embedding_key == "v_appearance":
                emb_t = dd.extract_v_appearance_only(model, video)[0]
            else:
                out = model(video, flows)
                emb_t = out[args.embedding_key][0]
        embeddings.append(emb_t.detach().cpu().numpy().astype(np.float32))

    return np.stack(embeddings, axis=0)


def count_valid_by_action_object(motions, objects, valid_openclose):
    action_counts = {}
    object_counts = {}
    for action in ("open", "close"):
        action_counts[action] = int(np.sum(valid_openclose & (motions == action)))

    valid_objects = objects[valid_openclose]
    unique_objects, counts = np.unique(valid_objects, return_counts=True)
    for obj, cnt in zip(unique_objects.tolist(), counts.tolist()):
        object_counts[str(obj)] = int(cnt)
    return action_counts, object_counts


def summarize_and_save(
    model_name,
    deltas_info,
    args,
    items,
    motions,
    objects,
    valid_openclose,
    video_ids,
    output_dir,
    extra_meta=None,
):
    deltas = deltas_info["deltas"]
    if deltas.size == 0:
        raise RuntimeError(
            f"No valid anchors with both hard/easy pairs for model={model_name}. "
            "Check detail_label distribution and --max-samples."
        )

    os.makedirs(output_dir, exist_ok=True)
    prefix = model_name.lower()
    out_png = os.path.join(output_dir, f"{prefix}_benchmark_b_delta_hist.png")
    out_json = os.path.join(output_dir, f"{prefix}_benchmark_b_delta_stats.json")
    out_npy = os.path.join(output_dir, f"{prefix}_benchmark_b_delta_values.npy")
    out_hard = os.path.join(output_dir, f"{prefix}_benchmark_b_hard_sim.npy")
    out_easy = os.path.join(output_dir, f"{prefix}_benchmark_b_easy_sim.npy")

    plot_delta_histogram(
        deltas=deltas,
        out_png=out_png,
        model_name=model_name,
        bins=args.bins,
        color=("#d62728" if model_name.lower() == "mae" else "#1f77b4"),
    )
    np.save(out_npy, deltas)
    np.save(out_hard, deltas_info["hard_sims"])
    np.save(out_easy, deltas_info["easy_sims"])

    action_counts, object_counts = count_valid_by_action_object(motions, objects, valid_openclose)
    anchor_ids = [video_ids[i] for i in deltas_info["anchor_indices"].tolist()]

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "data_json": args.data_json,
        "data_root": args.data_root,
        "seed": args.seed,
        "device": str(choose_device(args.device)),
        "reduction": args.reduction,
        "num_total_selected_samples": int(len(items)),
        "num_valid_openclose_samples": int(deltas_info["num_valid_openclose"]),
        "num_anchors_evaluated": int(len(deltas)),
        "num_anchors_without_pair": int(deltas_info["num_anchors_without_pair"]),
        "action_counts_valid": action_counts,
        "object_counts_valid": object_counts,
        "avg_hard_candidate_count": float(np.mean(deltas_info["hard_candidate_counts"])),
        "avg_easy_candidate_count": float(np.mean(deltas_info["easy_candidate_counts"])),
        "hard_sim_mean": float(np.mean(deltas_info["hard_sims"])),
        "easy_sim_mean": float(np.mean(deltas_info["easy_sims"])),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_min": float(np.min(deltas)),
        "delta_max": float(np.max(deltas)),
        "inversion_ratio_delta_lt_0": float(np.mean(deltas < 0.0)),
        "positive_ratio_delta_gt_0": float(np.mean(deltas > 0.0)),
        "anchor_video_ids": anchor_ids,
        "outputs": {
            "hist_png": out_png,
            "delta_npy": out_npy,
            "hard_sim_npy": out_hard,
            "easy_sim_npy": out_easy,
        },
    }
    if extra_meta:
        stats.update(extra_meta)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[DONE] {model_name} stats saved: {out_json}")
    print(json.dumps(stats, indent=2))
    return stats, deltas


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    md.set_seed(args.seed)
    dd.set_seed(args.seed)
    device = choose_device(args.device)
    print(f"[INFO] device: {device}")
    print(f"[INFO] seed: {args.seed}")
    print(f"[INFO] model_type: {args.model_type}")
    print(f"[INFO] reduction: {args.reduction}")
    print(f"[INFO] data_json: {args.data_json}")

    items = load_items(args.data_json, max_samples=args.max_samples)
    motions, objects, valid_openclose, video_ids = build_anchor_metadata(items)
    print(f"[INFO] selected samples: {len(items)}")
    print(f"[INFO] open/close-valid samples: {int(np.sum(valid_openclose))}")

    run_mae = args.model_type in {"mae", "both"}
    run_detach = args.model_type in {"detach", "both"}

    results = {}
    if run_mae:
        print("[INFO] extracting MAE embeddings...")
        mae_embeddings = extract_mae_embeddings(args=args, items=items, device=device)
        mae_delta_info = compute_delta_similarity(
            embeddings=mae_embeddings,
            motions=motions,
            objects=objects,
            valid_openclose=valid_openclose,
            reduction=args.reduction,
        )
        mae_stats, mae_deltas = summarize_and_save(
            model_name="MAE",
            deltas_info=mae_delta_info,
            args=args,
            items=items,
            motions=motions,
            objects=objects,
            valid_openclose=valid_openclose,
            video_ids=video_ids,
            output_dir=args.output_dir,
            extra_meta={
                "checkpoint_path": args.mae_checkpoint_path,
                "args_json": args.mae_args_json,
                "mae_num_frames": args.mae_num_frames,
                "mae_embedding_mode": args.mae_embedding_mode,
            },
        )
        results["MAE"] = {"stats": mae_stats, "deltas": mae_deltas}

    if run_detach:
        print("[INFO] extracting Detach embeddings...")
        detach_embeddings = extract_detach_embeddings(args=args, items=items, device=device)
        detach_delta_info = compute_delta_similarity(
            embeddings=detach_embeddings,
            motions=motions,
            objects=objects,
            valid_openclose=valid_openclose,
            reduction=args.reduction,
        )
        detach_stats, detach_deltas = summarize_and_save(
            model_name="Detach",
            deltas_info=detach_delta_info,
            args=args,
            items=items,
            motions=motions,
            objects=objects,
            valid_openclose=valid_openclose,
            video_ids=video_ids,
            output_dir=args.output_dir,
            extra_meta={
                "checkpoint_path": args.detach_checkpoint_path,
                "embedding_key": args.embedding_key,
                "num_frames": args.num_frames,
                "img_size": args.img_size,
            },
        )
        results["Detach"] = {"stats": detach_stats, "deltas": detach_deltas}

    if "MAE" in results and "Detach" in results:
        overlay_png = os.path.join(args.output_dir, "benchmark_b_delta_overlay_mae_vs_detach.png")
        plot_overlay(
            mae_deltas=results["MAE"]["deltas"],
            detach_deltas=results["Detach"]["deltas"],
            out_png=overlay_png,
            bins=args.bins,
        )
        compare_json = os.path.join(args.output_dir, "benchmark_b_delta_compare_mae_vs_detach.json")
        compare_stats = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_json": args.data_json,
            "num_selected_samples": int(len(items)),
            "num_valid_openclose_samples": int(np.sum(valid_openclose)),
            "mae_delta_mean": float(np.mean(results["MAE"]["deltas"])),
            "detach_delta_mean": float(np.mean(results["Detach"]["deltas"])),
            "mae_inversion_ratio_delta_lt_0": float(np.mean(results["MAE"]["deltas"] < 0.0)),
            "detach_inversion_ratio_delta_lt_0": float(np.mean(results["Detach"]["deltas"] < 0.0)),
            "outputs": {"overlay_png": overlay_png},
        }
        with open(compare_json, "w", encoding="utf-8") as f:
            json.dump(compare_stats, f, indent=2)
        print(f"[DONE] compare stats saved: {compare_json}")
        print(json.dumps(compare_stats, indent=2))

    print("[DONE] Benchmark B delta similarity finished.")


if __name__ == "__main__":
    main()
