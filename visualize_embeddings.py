
import os
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image
import cv2
from torchvision import transforms as T
import torch.nn.functional as F
from method import MethodLightningModule
from linear_probe import set_module_params, get_backbone_with_mode # Import utility functions

# Hardcoded defaults/paths (fallback)
DATA_ROOT = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/"
DEFAULT_CHECKPOINT = "/home/jaemo/Method/checkpoints/method/Opportunity++/noFlow.ckpt"
JSON_PATH = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/action/pretrain_cropped_with_flow.json"

def load_video_frames(video_path, num_frames=20, transform=None):
    """
    Load frames from a video file.
    """
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 1:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = np.zeros(num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        else:
            frames.append(Image.new('RGB', (224, 224)))
    cap.release()
    
    if transform:
        frames_tensor = transform(frames) # Using ClipConsistentTransforms or similar if available
    else:
        # Fallback manual transform
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        normalize = T.Normalize(mean=mean, std=std)
        frames_tensor = torch.stack([normalize(T.ToTensor()(T.Resize((224, 224), antialias=True)(f))) for f in frames])

    return frames_tensor

def load_flow_frames(flow_path, num_frames=20):
    """Load and process optical flow."""
    if flow_path is None or not os.path.exists(os.path.join(flow_path, "flow.npy")):
        return None
    try:
        flow = np.load(os.path.join(flow_path, "flow.npy"))
        flow = torch.from_numpy(flow).float() # [T_seq, 2, H, W]
        
        # Resize if needed
        T_seq, C, H, W = flow.shape
        if H != 224 or W != 224:
            flow = F.interpolate(flow, size=(224, 224), mode='bilinear', align_corners=False)
            
        # Sample frames
        if T_seq != num_frames:
             indices = np.linspace(0, T_seq - 1, num_frames, dtype=int)
             flow = flow[indices]
             
        return flow # [T, 2, H, W]
    except Exception as e:
        print(f"Error loading flow from {flow_path}: {e}")
        return None

from torch.utils.data import Dataset, DataLoader

class SimpleVideoDataset(Dataset):
    def __init__(self, data_list, num_frames=20, transform=None, use_flow=False, cache_dir=None):
        self.data_list = data_list
        self.num_frames = num_frames
        self.transform = transform
        self.use_flow = use_flow
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        video_rel_path = item['frame_path']
        video_full_path = os.path.join(DATA_ROOT, video_rel_path)
        
        frames = None
        
        # 1. Try Loading from Cache
        if self.cache_dir:
            parts = video_full_path.split(os.sep)
            if len(parts) >= 2:
                last_two_parts = '/'.join(parts[-2:])
                cache_subdir = os.path.join(self.cache_dir, "videos")
                cache_path = os.path.join(cache_subdir, last_two_parts)
                cache_path = cache_path.rsplit('.', 1)[0] + '.pt'
                
                if os.path.exists(cache_path):
                    try:
                        from PIL import Image
                        with torch.serialization.safe_globals({Image.Image}):
                            frames = torch.load(cache_path)
                    except Exception as e:
                        print(f"Failed to load cache {cache_path}: {e}")
                        frames = None

        # 2. If no cache or load failed, load from video
        if frames is None:
            frames = list_video_frames_pil(video_full_path, self.num_frames)
            
        # 3. Transform
        if frames is not None and self.transform is not None:
            try:
                frames_tensor = self.transform(frames)
            except Exception as e:
                print(f"Transform failed for {video_full_path}: {e}")
                frames_tensor = torch.zeros(self.num_frames, 3, 224, 224)
        else:
             frames_tensor = torch.zeros(self.num_frames, 3, 224, 224)

        flow_tensor = None
        if self.use_flow:
            flow_rel_path = item.get('optical_flow_dir')
            if flow_rel_path:
                flow_full_path = os.path.join(DATA_ROOT, flow_rel_path)
                f_tensor = load_flow_frames(flow_full_path, num_frames=self.num_frames)
                if f_tensor is not None:
                    flow_tensor = f_tensor
            
            if flow_tensor is None: 
                 # Even if use_flow is True, if we can't load, we must provide a tensor for batching stability
                 # BUT if the user strictly meant "None" for flows argument in model, that's batch-level.
                 # If we are here, we are item-level.
                 # Sticking to Zeros for missing flow if use_flow=True is safest for DataLoader.
                 # User likely meant passing None to the MODEL if use_flow is False. 
                 # Which I am already doing, but making the dataset return None makes it explicit.
                 flow_tensor = torch.zeros(self.num_frames, 2, 224, 224)

        return {
            "video": frames_tensor,
            "flow": flow_tensor,
            "label": item['label']
        }

def list_video_frames_pil(video_path, num_frames):
    if not os.path.exists(video_path):
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            frames.append(Image.fromarray(frame))
        else:
            if frames:
                frames.append(frames[-1].copy())
            else:
                from PIL import Image
                frames.append(Image.new('RGB', (224, 224)))
    cap.release()
    return frames

def custom_collate_fn(batch):
    videos = torch.stack([item['video'] for item in batch])
    labels = torch.as_tensor(np.array([item['label'] for item in batch]))
    
    flows = [item['flow'] for item in batch]
    if all(f is None for f in flows):
        flows = None
    else:
        # Handle case where some might be None if logic was mixed (though I ensured Tensors if use_flow=True)
        # Verify all are tensors
        if any(f is None for f in flows):
             # This happens if use_flow=True but I returned None for some? 
             # I put fallback to zeros above.
             # But if logic changes, let's be safe.
             valid_sample = next((f for f in flows if f is not None), None)
             if valid_sample is not None:
                 shape = valid_sample.shape
                 flows = [f if f is not None else torch.zeros(shape) for f in flows]
             else:
                 flows = None
        
        if flows is not None:
             flows = torch.stack(flows)
             
    return {'video': videos, 'label': labels, 'flow': flows}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    from pytorch_lightning import seed_everything
    seed_everything(42, workers=True)
    
    # Setup args for model loading
    if args.checkpoint_path is None:
        args.checkpoint_path = DEFAULT_CHECKPOINT
        
    # Helper to set model/dataset names from path
    args = set_module_params(args)
    
    print(f"Loading checkpoint: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    # Load Model
    from linear_probe import load_pretrained_model
    try:
        print(f"[Debug] Loading model with args: {vars(args)}")
        
        # --- Inspect Checkpoint Structure Code ---
        print(f"\n[Inspect] Loading checkpoint file directly: {args.checkpoint_path}")
        raw_ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        print(f"[Inspect] Checkpoint keys: {list(raw_ckpt.keys())}")
        if 'state_dict' in raw_ckpt:
            state_dict_keys = list(raw_ckpt['state_dict'].keys())
            print(f"[Inspect] Total parameters in state_dict: {len(state_dict_keys)}")
            print("[Inspect] Parameter group prefixes (first 2 levels):")
            prefixes = set()
            for k in state_dict_keys:
                parts = k.split('.')
                if len(parts) >= 2:
                    prefixes.add(f"{parts[0]}.{parts[1]}")
                else:
                    prefixes.add(parts[0])
            for p in sorted(list(prefixes)):
                print(f"  - {p}")
        print("-" * 40)
        # -----------------------------------------

        model = load_pretrained_model(args)
    except Exception as e:
        print(f"Failed to load using linear_probe utility: {e}")
        print("Falling back to direct loading...")
        model = MethodLightningModule.load_from_checkpoint(args.checkpoint_path, strict=False)
        
    model.to(device)
    model.to(device)
    # Important: main.py visualizes features extracted during training loop (Train Mode).
    # Eval mode uses running stats for BN, which might differ from batch stats used in training visualization.
    # To reproduce "Epoch X" plots exactly, we should use .train() mode (while keeping no_grad).
    print("Setting model to TRAIN mode for feature extraction (to match training-time visualization)...")
    # model.train()
    model.eval() # Previous behavior
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train()
    print("shared encoder net[1]: ", model.video_model.shared_encoder.net[1])
    # Identify video encoder
    if args.model_name == "method":
        video_encoder = model.video_model
    elif args.model_name == "mae":
        video_encoder = model.model
    else:
        video_encoder = getattr(model, 'video_model', None)
        if video_encoder is None:
             raise ValueError(f"Could not identify video encoder for model {args.model_name}")

    # 2. Load Data
    json_path = args.json_path if args.json_path else JSON_PATH
    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)['data']

    # Filter labels (All labels or specific subset?)
    # For pretraining/linear_train, let's use all distinct labels found, up to a limit for color palette
    target_labels = sorted(list(set(d['label'] for d in data)))
    # If too many labels, maybe limit? But t-SNE can handle it.
    
    import random
    filtered_data = [item for item in data if item['label'] in target_labels]
    
    if args.limit_samples:
        random.seed(42) 
        random.shuffle(filtered_data)
        filtered_data = filtered_data[:args.limit_samples]
        
    print(f"Found {len(filtered_data)} samples.")

    # Transform setup
    from datamodule import ClipConsistentTransforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = ClipConsistentTransforms(size=(224, 224), mean=mean, std=std)
    
    # Dataset & DataLoader
    cache_dir = os.path.join(DATA_ROOT, "caches")
    print(f"Using cache dir: {cache_dir}")
    dataset = SimpleVideoDataset(filtered_data, num_frames=args.num_frames, transform=transform, use_flow=args.use_flow, cache_dir=cache_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # Process items
    print("Extracting embeddings...")
    z_embeddings = []
    app_embeddings = []
    mot_embeddings = []
    labels = []

    model.train()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            _ = video_encoder(batch['video'].to(device), None)
            if i == 200:
                break
    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0: 
            print(f"Processing batch {batch_idx}/{len(dataloader)}...", end='\r')
        
        videos = batch['video'].to(device) # [B, T, C, H, W]
        
        flow_input = None
        if batch['flow'] is not None:
             flow_input = batch['flow'].to(device)

        with torch.no_grad():
            if args.model_name == "method":
                # Ensure we pass flow_input which is either Tensor or None
                out = video_encoder(videos, flows=flow_input)
                
                # Manually reconstruct z_video_online as requested
                v_app = out["v_appearance"]
                v_mot = out["v_motion"]
                
                # Normalize individual components
                v_app_norm = F.normalize(v_app, dim=1)
                v_mot_norm = F.normalize(v_mot, dim=1)
                
                # Concatenate (Simulating z_video_online construction)
                z_emb_reconstructed = torch.cat([v_app_norm, v_mot_norm], dim=1)
                
                # Normalize the combined vector for t-SNE (consistent with reference)
                z_emb = F.normalize(z_emb_reconstructed, dim=1).cpu().numpy()
                
                app_emb = v_app_norm.cpu().numpy()
                mot_emb = v_mot_norm.cpu().numpy()
            elif args.model_name == "mae":
                 # Placeholder
                 pass
            
            z_embeddings.append(z_emb)
            app_embeddings.append(app_emb)
            mot_embeddings.append(mot_emb)
            labels.extend(batch['label'].numpy())
            
    if not z_embeddings:
        print("No embeddings extracted.")
        return

    z_embeddings = np.concatenate(z_embeddings, axis=0)
    app_embeddings = np.concatenate(app_embeddings, axis=0)
    mot_embeddings = np.concatenate(mot_embeddings, axis=0)
    labels = np.array(labels)
    
    print(f"\nExtracted shapes: Z={z_embeddings.shape}, App={app_embeddings.shape}, Mot={mot_embeddings.shape}")

    # 3. Visualization Helper
    def run_tsne_and_plot(emb_data, title_suffix, filename_suffix):
        print(f"Running t-SNE for {title_suffix}...")
        n_samples = len(emb_data)
        perplexity = min(30, n_samples - 1)
        # Use metric='cosine' to match visualizes.py
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto', metric='cosine')
        embeddings_2d = tsne.fit_transform(emb_data)

        plt.figure(figsize=(12, 10))
        
        class_dic = {
                0: 'Open Door 1', 1: 'Open Door 2', 2: 'Close Door 1', 3: 'Close Door 2',
                4: 'Open Fridge', 5: 'Close Fridge', 6: 'Open Dishwasher', 7: 'Close Dishwasher',
                8: 'Open Drawer 1', 9: 'Close Drawer 1', 10: 'Open Drawer 2',
                11: 'Close Drawer 2', 12: 'Open Drawer 3', 13: 'Close Drawer 3'
            }
        plot_labels = [class_dic.get(l, str(l)) for l in labels]
        unique_labels = sorted(list(set(plot_labels)))
        colors = sns.color_palette("tab10", len(unique_labels)) if len(unique_labels) <= 10 else sns.color_palette("husl", len(unique_labels))

        sns.scatterplot(
            x=embeddings_2d[:, 0], 
            y=embeddings_2d[:, 1], 
            hue=plot_labels, 
            palette=colors,
            s=40,
            alpha=0.7
        )
        
        plt.title(f"t-SNE {title_suffix} (Model: {args.model_name})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Class")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        output_filename = f"tsne_{args.model_name}_{'flow' if args.use_flow else 'noflow'}_{filename_suffix}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved {title_suffix} to {output_path}")
        plt.close()

    # Run for all 3
    run_tsne_and_plot(z_embeddings, "Z_Video (Combined)", "z_video")
    run_tsne_and_plot(app_embeddings, "Appearance Only", "appearance")
    run_tsne_and_plot(mot_embeddings, "Motion Only", "motion")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="method")
    parser.add_argument("--dataset_name", type=str, default="Opportunity++") 
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--use_flow", action='store_true')
    parser.add_argument("--limit_samples", type=int, default=None)
    
    args = parser.parse_args()
    main(args)
