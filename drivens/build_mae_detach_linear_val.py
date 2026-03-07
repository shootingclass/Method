import argparse
import json
import os
import shlex
import subprocess
import sys


DEFAULT_DATA_ROOT = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window"
DEFAULT_DATA_JSON = os.path.join(DEFAULT_DATA_ROOT, "action/linear_val.json")

DEFAULT_MAE_SCRIPT = "/home/jaemo/Method/drivens/mae_driven.py"
DEFAULT_DETACH_SCRIPT = "/home/jaemo/Method/drivens/detach_driven.py"

DEFAULT_MAE_CKPT = "/home/jaemo/Method/checkpoints/evimae/Opportunity++/models/evi_model.299.pth"
DEFAULT_MAE_ARGS = "/home/jaemo/Method/checkpoints/evimae/Opportunity++/args.json"
DEFAULT_DETACH_CKPT = "/home/jaemo/Method/checkpoints/method/Opportunity++/last.ckpt"

DEFAULT_MAE_OUT = "/home/jaemo/Method/drivens/mae_linear_val"
DEFAULT_DETACH_OUT = "/home/jaemo/Method/drivens/detach_linear_val"

OPP_CLASS_NAMES = {
    0: "Open_Door_1",
    1: "Open_Door_2",
    2: "Close_Door_1",
    3: "Close_Door_2",
    4: "Open_Fridge",
    5: "Close_Fridge",
    6: "Open_Dishwasher",
    7: "Close_Dishwasher",
    8: "Open_Drawer_1",
    9: "Close_Drawer_1",
    10: "Open_Drawer_2",
    11: "Close_Drawer_2",
    12: "Open_Drawer_3",
    13: "Close_Drawer_3",
}


def load_items(data_json):
    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("data", [])
    if not items:
        raise ValueError(f"No samples found in {data_json}")
    return items


def resolve_indices(items, sample_indices, class_labels, max_samples):
    if sample_indices:
        indices = [i for i in sample_indices if 0 <= i < len(items)]
    else:
        indices = list(range(len(items)))

    if class_labels:
        labels = set(class_labels)
        indices = [i for i in indices if items[i].get("label") in labels]

    if max_samples > 0:
        indices = indices[:max_samples]

    return indices


def build_output_dir(root_dir, item, sample_index):
    label = item.get("label")
    class_name = OPP_CLASS_NAMES.get(label, f"Class_{label}" if label is not None else "Unknown")
    class_dir = f"{label}_{class_name}_all" if label is not None else "unknown_label_all"
    video_id = item.get("video_id", f"sample_{sample_index}")
    sample_dir = f"{sample_index}_{video_id}"
    return os.path.join(root_dir, class_dir, sample_dir)


def run_command(cmd, dry_run):
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"[RUN] {pretty}")
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def maybe_run_mae(args, sample_index, mae_out_dir, skip_existing):
    result_json = os.path.join(mae_out_dir, "run_once_result.json")
    if skip_existing and os.path.exists(result_json):
        print(f"[SKIP][mae] exists: {result_json}")
        return 0

    os.makedirs(mae_out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        args.mae_script,
        "--checkpoint-path",
        args.mae_checkpoint_path,
        "--args-json",
        args.mae_args_json,
        "--data-json",
        args.data_json,
        "--sample-index",
        str(sample_index),
        "--output-dir",
        mae_out_dir,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    return run_command(cmd, dry_run=args.dry_run)


def maybe_run_detach(args, sample_index, detach_out_dir, skip_existing):
    result_json = os.path.join(detach_out_dir, "run_once_result.json")
    if skip_existing and os.path.exists(result_json):
        print(f"[SKIP][detach] exists: {result_json}")
        return 0

    os.makedirs(detach_out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        args.detach_script,
        "--checkpoint-path",
        args.detach_checkpoint_path,
        "--data-root",
        args.data_root,
        "--data-json",
        args.data_json,
        "--sample-index",
        str(sample_index),
        "--output-dir",
        detach_out_dir,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    return run_command(cmd, dry_run=args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MAE vs Detach qualitative outputs from linear_val.json"
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-json", type=str, default=DEFAULT_DATA_JSON)

    parser.add_argument("--mae-script", type=str, default=DEFAULT_MAE_SCRIPT)
    parser.add_argument("--detach-script", type=str, default=DEFAULT_DETACH_SCRIPT)

    parser.add_argument("--mae-checkpoint-path", type=str, default=DEFAULT_MAE_CKPT)
    parser.add_argument("--mae-args-json", type=str, default=DEFAULT_MAE_ARGS)
    parser.add_argument("--detach-checkpoint-path", type=str, default=DEFAULT_DETACH_CKPT)

    parser.add_argument("--mae-output-root", type=str, default=DEFAULT_MAE_OUT)
    parser.add_argument("--detach-output-root", type=str, default=DEFAULT_DETACH_OUT)

    parser.add_argument("--sample-indices", nargs="*", type=int, default=[])
    parser.add_argument("--class-labels", nargs="*", type=int, default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--only-mae", action="store_true")
    parser.add_argument("--only-detach", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.only_mae and args.only_detach:
        raise ValueError("Use only one of --only-mae / --only-detach.")

    run_mae = not args.only_detach
    run_detach = not args.only_mae
    skip_existing = not args.force

    items = load_items(args.data_json)
    indices = resolve_indices(
        items=items,
        sample_indices=args.sample_indices,
        class_labels=args.class_labels,
        max_samples=args.max_samples,
    )

    print(f"[INFO] total selected samples: {len(indices)}")
    if not indices:
        return

    failures = []
    for i in indices:
        if i < 0 or i >= len(items):
            failures.append((i, "sample_index out of range"))
            continue

        item = items[i]
        mae_out_dir = build_output_dir(args.mae_output_root, item, i)
        detach_out_dir = build_output_dir(args.detach_output_root, item, i)
        print(f"\n[INFO] sample_index={i} label={item.get('label')} video_id={item.get('video_id')}")

        if run_mae:
            code = maybe_run_mae(args, i, mae_out_dir, skip_existing)
            if code != 0:
                failures.append((i, f"mae failed with exit code {code}"))
                continue

        if run_detach:
            code = maybe_run_detach(args, i, detach_out_dir, skip_existing)
            if code != 0:
                failures.append((i, f"detach failed with exit code {code}"))
                continue

    if failures:
        print("\n[FAILED]")
        for sample_index, reason in failures:
            print(f"  - sample_index={sample_index}: {reason}")
        raise SystemExit(1)

    print("\n[DONE] All selected samples processed.")


if __name__ == "__main__":
    main()
