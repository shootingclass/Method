# -*- coding: utf-8 -*-
# Run script for sequence-level fine-tuning (HWU dataset).
# Uses EVISequenceDataset for sequence-format JSON,
# and train_seq for window-level encoding + mean pooling.

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from dataloader_seq import EVISequenceDataset, collate_sequence
import models, random
import numpy as np
import warnings

from traintest_ft_seq import train_seq, validate_seq, test_seq

# Set batch size for DataLoader to match number of GPUs (1 sequence per GPU)
train_batch_size = max(1, torch.cuda.device_count())
print(f"Setting DataLoader batch size to {train_batch_size} (1 per GPU)")

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json (sequence format)")
parser.add_argument("--data-val", type=str, default='', help="validation data json (sequence format)")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json (sequence format)")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=5, help="number of classes")
parser.add_argument("--model", type=str, default='evi-mae-ft', help="the model used")
parser.add_argument("--dataset", type=str, default="hwu", help="the dataset used", choices=["opp", "hwu"])
parser.add_argument("--noise", help='if use noise augmentation', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (number of sequences)')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading')
parser.add_argument("--n-epochs", type=int, default=200, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if metric doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="acc", help="the main evaluation metrics", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=60, type=int, help="when to start decay")
parser.add_argument("--lrscheduler_step", default=60, type=int, help="the number of step to decrease the learning rate")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa", help='if do weight averaging', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=40, help="which epoch to start weight averaging")
parser.add_argument("--wa_end", type=int, default=200, help="which epoch to end weight averaging")
parser.add_argument("--wa_num", type=int, default=12, help="how many epochs to average")

parser.add_argument("--only_val", help='if only do evaluation', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

parser.add_argument("--head_lr", type=float, default=100.0, help="learning rate ratio for newly initialized layers")
parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval, default='False')

parser.add_argument("--base_lr", type=float, default=1, help="the base learning rate of the model")
parser.add_argument("--image_as_video", help='if use image as video', type=ast.literal_eval, default='False')
parser.add_argument("--use_checkpoint", help='if use gradient checkpointing', type=ast.literal_eval, default='True')

# imu
parser.add_argument("--imu_target_length", type=int, default=100, help="the target length of imu data")
parser.add_argument("--imu_plot_type", type=str, default='stft', help="the plot type of imu data", choices=['fbank', 'rp', 'mel', 'raw', 'stft'])
parser.add_argument("--imu_plot_height", type=int, default=128, help="the plot height of imu data")
parser.add_argument("--imu_patch_size", type=int, default=16, help="the patch size of imu data")
parser.add_argument("--imu_dataset_mean", type=str, help="the dataset imu mean")
parser.add_argument("--imu_dataset_std", type=str, help="the dataset imu std")
parser.add_argument("--imu_channel_num", type=int, default=6, help="the channel number of imu data")
parser.add_argument("--imu_encoder_embed_dim", type=int, default=768, help="the embed dim of imu encoder")
parser.add_argument("--imu_encoder_depth", type=int, default=11, help="the depth of imu encoder")
parser.add_argument("--imu_encoder_num_heads", type=int, default=12, help="the num heads of imu encoder")
parser.add_argument("--imu_enable_graph", type=ast.literal_eval, default='False', help="enable graph for imu data")
parser.add_argument("--imu_graph_net", type=str, default='gin', help="the graph net for imu data", choices=['gin'])
parser.add_argument("--imu_two_stream", type=ast.literal_eval, default='False', help="if use two mlp")

# video
parser.add_argument("--video_img_size", type=int, default=224, help="the image size of video data")
parser.add_argument("--video_patch_size", type=int, default=16, help="the patch size of video data")
parser.add_argument("--video_encoder_num_classes", type=int, default=0, help="the number of classes of video encoder")
parser.add_argument("--video_decoder_num_classes", type=int, default=1536, help="the number of classes of video decoder")
parser.add_argument("--video_mlp_ratio", type=int, default=4, help="the mlp ratio of video data")
parser.add_argument("--video_qkv_bias", type=ast.literal_eval, default='True', help="the qkv bias of video data")
parser.add_argument("--video_encoder_embed_dim", type=int, default=384, help="the embed dim of video encoder")
parser.add_argument("--video_encoder_depth", type=int, default=12, help="the depth of video encoder")
parser.add_argument("--video_encoder_num_heads", type=int, default=6, help="the num heads of video encoder")
parser.add_argument("--video_decoder_embed_dim", type=int, default=192, help="the embed dim of video decoder")
parser.add_argument("--video_decoder_num_heads", type=int, default=3, help="the num heads of video decoder")
parser.add_argument("--video_masking_ratio", type=float, default=0.9, help="video masking ratio")

parser.add_argument("--rseed", type=int, default=42, help="random seed")

args = parser.parse_args()

# set random seed
seed = args.rseed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# ============================================================
# IMU configs (same structure as run_evimae_ft.py)
# ============================================================
imu_conf = {
    'num_mel_bins': 128,
    'target_length': args.imu_target_length,
    'freqm': args.freqm,
    'timem': args.timem,
    'mixup': args.mixup,
    'dataset': args.dataset,
    'mode': 'train',
    'mean': args.imu_dataset_mean,
    'std': args.imu_dataset_std,
    'noise': args.noise,
    'label_smooth': args.label_smooth,
    'im_res': args.video_img_size,
    'imu_channel_num': args.imu_channel_num,
}
val_imu_conf = {
    'num_mel_bins': 128,
    'target_length': args.imu_target_length,
    'freqm': 0,
    'timem': 0,
    'mixup': 0,
    'dataset': args.dataset,
    'mode': 'eval',
    'mean': args.imu_dataset_mean,
    'std': args.imu_dataset_std,
    'noise': False,
    'im_res': args.video_img_size,
    'imu_channel_num': args.imu_channel_num,
}

# ============================================================
# Shared class mapping (auto-populated from train set, reused by val/test)
# ============================================================
class_name_to_idx = {}

# ============================================================
# Sequence DataLoaders
# ============================================================
print('Loading sequence dataset for training...')
train_dataset = EVISequenceDataset(
    args.data_train, imu_conf=imu_conf, label_csv=args.label_csv,
    video_masking_ratio=args.video_masking_ratio, image_as_video=args.image_as_video,
    class_name_to_idx=class_name_to_idx
)
train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True, drop_last=False,
    collate_fn=collate_sequence
)

print('Loading sequence dataset for validation...')
val_dataset = EVISequenceDataset(
    args.data_val, imu_conf=val_imu_conf, label_csv=args.label_csv,
    video_masking_ratio=args.video_masking_ratio,
    class_name_to_idx=class_name_to_idx
)
val_loader = DataLoader(
    val_dataset, batch_size=train_batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, drop_last=False,
    collate_fn=collate_sequence
)

if args.data_eval is not None:
    print('Loading sequence dataset for evaluation...')
    eval_dataset = EVISequenceDataset(
        args.data_eval, imu_conf=val_imu_conf, label_csv=args.label_csv,
        video_masking_ratio=args.video_masking_ratio,
        class_name_to_idx=class_name_to_idx
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=train_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_sequence
    )
else:
    eval_loader = None

# Populate class names for confusion matrix logging
# User requested hardcoded class names instead of loading from CSV
if args.dataset == 'hwu':
    args.class_names = ['tidy', 'dishes', 'sandwich', 'cereals', 'tea']
elif args.dataset == 'opp':
    # Opportunity dataset classes
    args.class_names = [
        'Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2',
        'Open Fridge', 'Close Fridge', 'Open Dishwasher', 'Close Dishwasher',
        'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
        'Close Drawer 2', 'Open Drawer 3', 'Close Drawer 3'
    ]
else:
    # Fallback
    raise ValueError(f"Unknown dataset: {args.dataset}")

print(f"Using class names: {args.class_names}")
print(f'Class mapping: {class_name_to_idx}')

# ============================================================
# Model
# ============================================================
if args.model == 'evi-mae-ft':
    video_model_dict = {
        'img_size': args.video_img_size,
        'patch_size': args.video_patch_size,
        'encoder_embed_dim': args.video_encoder_embed_dim,
        'encoder_depth': args.video_encoder_depth,
        'encoder_num_heads': args.video_encoder_num_heads,
        'mlp_ratio': args.video_mlp_ratio,
        'qkv_bias': args.video_qkv_bias,
        'encoder_num_classes': args.video_encoder_num_classes,
        'decode_num_classes': args.video_decoder_num_classes,
        'decode_embed_dim': args.video_decoder_embed_dim,
        'decode_num_heads': args.video_decoder_num_heads,
    }
    imu_model_dict = {
        'target_length': args.imu_target_length,
        'plot_type': args.imu_plot_type,
        'plot_height': args.imu_plot_height,
        'patch_size': args.imu_patch_size,
        'channel_num': args.imu_channel_num,
        'encoder_embed_dim': args.imu_encoder_embed_dim,
        'encoder_depth': args.imu_encoder_depth,
        'encoder_num_heads': args.imu_encoder_num_heads,
        'enable_graph': args.imu_enable_graph,
        'imu_graph_net': args.imu_graph_net,
        'imu_two_stream': args.imu_two_stream,
    }
    evi_model = models.EVIMAEFT(label_dim=args.n_class, video_model_dict=video_model_dict, imu_model_dict=imu_model_dict, use_checkpoint=args.use_checkpoint)
else:
    raise ValueError('model not supported')

# ============================================================
# Load pretrained weights
# ============================================================
if args.pretrain_path == 'None':
    warnings.warn("Note you are finetuning a model without any pretraining.")

if args.pretrain_path != 'None':
    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    
    # For sequence training we do NOT use DataParallel
    # Check if weights have "module." prefix
    cleaned_weight = {}
    for k, v in mdl_weight.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_weight[name] = v
    
    model_state_dict = evi_model.state_dict()
    filtered_state_dict = {}
    
    print('now load evi-mae pretrained weights from ', args.pretrain_path)
    for name, param in cleaned_weight.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                filtered_state_dict[name] = param
            else:
                print(f"Skipped parameter (shape mismatch): {name} - mdl_weight: {param.shape}, evi_model: {model_state_dict[name].shape}")
    
    miss, unexpected = evi_model.load_state_dict(filtered_state_dict, strict=False)
    print('miss', miss)
    print('unexpected', unexpected)

# Wrap model with DataParallel for multi-GPU support
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    evi_model = torch.nn.DataParallel(evi_model)
evi_model = evi_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ============================================================
# Create experiment directory
# ============================================================
print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
with open(args.exp_dir + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# ============================================================
# Train
# ============================================================
print('Now starting sequence-level training for {:d} epochs.'.format(args.n_epochs))
train_seq(evi_model, train_loader, val_loader, args)

# ============================================================
# Final evaluation
# ============================================================
if eval_loader is not None:
    print('start final evaluation on test set')
    best_model_path = os.path.join(args.exp_dir, "models", "best_evi_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        state_dict = torch.load(best_model_path, map_location='cpu')
        # Remove 'module.' prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            cleaned[name] = v
        evi_model.load_state_dict(cleaned, strict=False)
    else:
        print("Best model not found, using current model state.")

    test_seq(evi_model, eval_loader, args)
