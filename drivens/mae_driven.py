import argparse
import json
import os
import random
import sys
import types
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIMAE_SRC = PROJECT_ROOT / "baseline_modules" / "evi-mae" / "src"

if str(EVIMAE_SRC) not in sys.path:
    sys.path.insert(0, str(EVIMAE_SRC))

try:
    import dataloader as evimae_dataloader  # noqa: E402
    import models as evimae_models  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Failed to import baseline_modules/evi-mae/src modules. "
        "Please run with the env that has pandas/decord/timm/torchaudio installed."
    ) from exc


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_dataset_deterministic(dataset, img_size=224):
    # 1) Disable random multi-scale crop -> center crop only.
    from transforms import GroupCenterCrop, GroupNormalize, Stack, ToTorchFormatTensor

    normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dataset.video_transform.transform = tv_transforms.Compose(
        [
            GroupCenterCrop(img_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ]
    )

    # 2) Disable random temporal clip sampling -> fixed center clip.
    def _fixed_sample_train_indices(self, num_frames):
        if num_frames - self.skip_length + 1 > 0:
            offset = (num_frames - self.skip_length) // 2
        else:
            offset = 0
        offsets = np.array([offset], dtype=int) + 1
        skip_offsets = np.zeros(self.skip_length // self.new_step, dtype=int)
        return offsets, skip_offsets

    dataset._sample_train_indices = types.MethodType(_fixed_sample_train_indices, dataset)


def build_model_dicts(args_dict):
    video_model_dict = {
        "img_size": args_dict["video_img_size"],
        "patch_size": args_dict["video_patch_size"],
        "encoder_embed_dim": args_dict["video_encoder_embed_dim"],
        "encoder_depth": args_dict["video_encoder_depth"],
        "encoder_num_heads": args_dict["video_encoder_num_heads"],
        "mlp_ratio": args_dict["video_mlp_ratio"],
        "qkv_bias": args_dict["video_qkv_bias"],
        "encoder_num_classes": args_dict["video_encoder_num_classes"],
        "decode_num_classes": args_dict["video_decoder_num_classes"],
        "decode_embed_dim": args_dict["video_decoder_embed_dim"],
        "decode_num_heads": args_dict["video_decoder_num_heads"],
        "masking_ratio": args_dict["video_masking_ratio"],
        "pretrain_modality": args_dict["pretrain_modality"],
    }

    imu_model_dict = {
        "target_length": args_dict["imu_target_length"],
        "masking_ratio": args_dict["imu_masking_ratio"],
        "mask_mode": args_dict["imu_mask_mode"],
        "plot_type": args_dict["imu_plot_type"],
        "plot_height": args_dict["imu_plot_height"],
        "patch_size": args_dict["imu_patch_size"],
        "channel_num": args_dict["imu_channel_num"],
        "encoder_embed_dim": args_dict["imu_encoder_embed_dim"],
        "encoder_depth": args_dict["imu_encoder_depth"],
        "encoder_num_heads": args_dict["imu_encoder_num_heads"],
        "enable_graph": args_dict["imu_enable_graph"],
        "imu_graph_net": args_dict["imu_graph_net"],
        "imu_graph_masking_ratio": args_dict["imu_graph_masking_ratio"],
    }
    return video_model_dict, imu_model_dict


def build_imu_conf(args_dict):
    return {
        "num_mel_bins": 128,
        "target_length": args_dict["imu_target_length"],
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "dataset": args_dict["dataset"],
        "mode": "eval",
        "mean": args_dict["imu_dataset_mean"],
        "std": args_dict["imu_dataset_std"],
        "noise": False,
        "label_smooth": 0,
        "im_res": args_dict["video_img_size"],
    }


def load_checkpoint_state_dict(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format at {ckpt_path}: {type(raw)}")

    cleaned = {}
    for key, value in raw.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] checkpoint loaded: {ckpt_path}")
    print(f"[INFO] missing keys: {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[INFO] missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"[INFO] unexpected (first 10): {unexpected[:10]}")


def ensure_dim(tensor, target_dim):
    while tensor.dim() < target_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def save_first_frame(video_input, save_path):
    # video_input: [1, 3, T, H, W], ImageNet normalized
    mean = torch.tensor([0.485, 0.456, 0.406], device=video_input.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=video_input.device).view(1, 3, 1, 1)

    frame = video_input[0, :, 0, :, :].unsqueeze(0)
    frame = (frame * std + mean).clamp(0, 1)
    frame_np = frame[0].permute(1, 2, 0).detach().cpu().numpy()
    frame_uint8 = (frame_np * 255.0).astype(np.uint8)
    Image.fromarray(frame_uint8).save(save_path)


def _save_original_frames(video_input, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    frames = _denorm_video_frames(video_input)
    for t, frame in enumerate(frames):
        frame_u8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(frame_u8).save(os.path.join(save_dir, f"frame_{t:03d}.png"))
    return {"original_frame_dir": save_dir, "num_frames": int(frames.shape[0])}


def _extract_video_tokens_nomask(model, video_input):
    # IMPORTANT:
    # We bypass forward_encoder because random_masking_unstructured shuffles token order
    # even when mask_ratio is 0, which breaks spatial alignment of heatmaps.
    with torch.no_grad():
        v = model.patch_embed_video(video_input)
        v = v + model.pos_embed_video.type_as(v).to(v.device).clone().detach()
        v = v + model.modality_video
        for blk in model.blocks_video:
            v = blk(v)
    return v


def _denorm_video_frames(video_input):
    # video_input: [1, 3, T, H, W], ImageNet normalized
    mean = torch.tensor([0.485, 0.456, 0.406], device=video_input.device).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=video_input.device).view(3, 1, 1, 1)

    frames = (video_input[0] * std + mean).clamp(0, 1)  # [3, T, H, W]
    frames = frames.permute(1, 2, 3, 0).detach().cpu().numpy()  # [T, H, W, 3]
    return frames


def _make_attention_overlays(model, imu_input, video_input, mask_mode, save_dir):
    del imu_input, mask_mode
    video_tokens = _extract_video_tokens_nomask(model=model, video_input=video_input)
    token_importance = video_tokens.pow(2).mean(dim=-1)  # [B, N]

    frame_h, frame_w = video_input.shape[-2], video_input.shape[-1]
    patch_h = frame_h // model.video_patch_size
    patch_w = frame_w // model.video_patch_size
    spatial_per_tube = patch_h * patch_w

    n_tokens = token_importance.shape[1]
    tubelets = n_tokens // spatial_per_tube
    maps = token_importance[0].reshape(tubelets, patch_h, patch_w).detach().cpu().numpy()

    t_input = video_input.shape[2]
    repeat_factor = max(1, int(model.video_tubelet_size))
    maps = np.repeat(maps, repeat_factor, axis=0)
    if maps.shape[0] < t_input:
        pad = np.repeat(maps[-1:,:,:], t_input - maps.shape[0], axis=0)
        maps = np.concatenate([maps, pad], axis=0)
    maps = maps[:t_input]

    os.makedirs(save_dir, exist_ok=True)
    frames = _denorm_video_frames(video_input)

    # Clip-level normalization for temporal consistency.
    flat_vals = maps.reshape(-1)
    low, high = np.percentile(flat_vals, [20, 95])
    if high - low < 1e-8:
        low, high = float(flat_vals.min()), float(flat_vals.max())

    for t in range(t_input):
        heat = maps[t]
        heat = np.clip((heat - low) / (high - low + 1e-8), 0.0, 1.0)
        heat = np.power(heat, 1.2)

        heat_img = Image.fromarray((heat * 255.0).astype(np.uint8)).resize((frame_w, frame_h), Image.BICUBIC)
        heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=2.0))
        heat_up = np.asarray(heat_img).astype(np.float32) / 255.0

        heat_rgb = mpl.colormaps["jet"](heat_up)[..., :3]

        frame = frames[t]
        overlay = np.clip(0.65 * frame + 0.35 * heat_rgb, 0.0, 1.0)
        overlay_u8 = (overlay * 255.0).astype(np.uint8)
        Image.fromarray(overlay_u8).save(os.path.join(save_dir, f"frame_{t:03d}.png"))

    return {
        "feature_overlay_dir": save_dir,
        "num_frames": int(t_input),
        "patch_grid": [int(patch_h), int(patch_w)],
        "tubelets": int(tubelets),
        "map_type": "video_token_energy_no_mask",
    }


def run_once(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device: {device}")
    print(f"[INFO] seed: {args.seed}")

    args_dict = load_json(args.args_json)
    video_model_dict, imu_model_dict = build_model_dicts(args_dict)

    model = evimae_models.EVIMAE(
        norm_pix_loss=args_dict["norm_pix_loss"],
        tr_pos=args_dict["tr_pos"],
        video_model_dict=video_model_dict,
        imu_model_dict=imu_model_dict,
    ).to(device)
    model.eval()

    load_checkpoint_state_dict(model, args.checkpoint_path)

    data_json = args.data_json if args.data_json else args_dict["data_train"]
    imu_conf = build_imu_conf(args_dict)
    dataset = evimae_dataloader.EVIDataset(
        data_json,
        imu_conf=imu_conf,
        label_csv=args_dict.get("label_csv"),
        video_masking_ratio=args_dict["video_masking_ratio"],
        image_as_video=args_dict.get("image_as_video", False),
    )
    set_dataset_deterministic(dataset, img_size=args_dict["video_img_size"])
    print(f"[INFO] dataset size: {len(dataset)}")

    sample = dataset[args.sample_index]
    imu_input, video_input, third, _label = sample

    imu_input = ensure_dim(imu_input, 4).to(device)
    video_input = ensure_dim(video_input, 5).to(device)

    if torch.is_tensor(third):
        v_masks = ensure_dim(third, 2).to(device)
        sample_meta = "mask_tensor"
    else:
        num_patches = model.patch_embed_video.num_patches
        v_masks = torch.zeros((imu_input.size(0), num_patches), dtype=torch.float32, device=device)
        sample_meta = str(third)

    with torch.no_grad():
        loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_g = model(
            imu_input,
            video_input,
            v_masks,
            mae_loss_weight=args_dict["mae_loss_weight"],
            contrast_loss_weight=args_dict["contrast_loss_weight"],
            mask_mode=args_dict["imu_mask_mode"],
        )

    os.makedirs(args.output_dir, exist_ok=True)
    first_frame_path = os.path.join(args.output_dir, "first_frame.png")
    save_first_frame(video_input, first_frame_path)
    original_info = _save_original_frames(
        video_input=video_input,
        save_dir=os.path.join(args.output_dir, "original_frames"),
    )
    attn_dir = os.path.join(args.output_dir, "attention_overlay")
    attn_info = _make_attention_overlays(
        model=model,
        imu_input=imu_input,
        video_input=video_input,
        mask_mode=args_dict["imu_mask_mode"],
        save_dir=attn_dir,
    )

    result = {
        "sample_index": args.sample_index,
        "seed": args.seed,
        "sample_meta": sample_meta,
        "imu_shape": list(imu_input.shape),
        "video_shape": list(video_input.shape),
        "v_mask_shape": list(v_masks.shape),
        "loss": float(loss.item()),
        "loss_mae": float(loss_mae.item()),
        "loss_mae_a": float(loss_mae_a.item()),
        "loss_mae_v": float(loss_mae_v.item()),
        "loss_c": float(loss_c.item()),
        "contrast_acc": float(c_acc.item()),
        "loss_g": float(loss_g.item()),
        "mask_a_shape": list(mask_a.shape) if mask_a is not None else None,
        "mask_v_shape": list(mask_v.shape) if mask_v is not None else None,
        "first_frame_path": first_frame_path,
        "original_info": original_info,
        "attention_info": attn_info,
    }

    result_path = os.path.join(args.output_dir, "run_once_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[DONE] single forward finished")
    print(json.dumps(result, indent=2))
    print(f"[DONE] saved result: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-run EVI-MAE driver")
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
        default="",
        help="Optional override for dataset json. If empty, args.json:data_train is used.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/jaemo/Method/drivens/mae",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_once(parse_args())
