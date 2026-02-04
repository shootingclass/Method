import torch
from method import MethodLightningModule

CHECKPOINT_PATH = "/home/jaemo/Method/checkpoints/method/Opportunity++/noFlow.ckpt"

def check_keys():
    print(f"Checking checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    print("\n--- Hyperparameters ---")
    if "hyper_parameters" in checkpoint:
        for k, v in checkpoint["hyper_parameters"].items():
            print(f"{k}: {v}")
    else:
        print("No hyperparameters found in checkpoint.")

    print("\n--- Model vs Checkpoint Keys ---")
    model = MethodLightningModule.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    missing_in_model = ckpt_keys - model_keys
    missing_in_ckpt = model_keys - ckpt_keys
    
    print(f"\nKeys in Checkpoint but NOT in Model (Ignored): {len(missing_in_model)}")
    for k in sorted(list(missing_in_model))[:10]:
        print(f"  {k}")
    if len(missing_in_model) > 10: print("  ...")
        
    print(f"\nKeys in Model but NOT in Checkpoint (Randomly Initialized): {len(missing_in_ckpt)}")
    for k in sorted(list(missing_in_ckpt))[:10]:
        print(f"  {k}")
    if len(missing_in_ckpt) > 10: print("  ...")
    
    # Check specific critical prefixes
    print("\n--- Critical Prefix Check (video_model) ---")
    video_keys_ckpt = [k for k in ckpt_keys if "video_model" in k]
    video_keys_model = [k for k in model_keys if "video_model" in k]
    print(f"Video Model Keys in Ckpt: {len(video_keys_ckpt)}")
    print(f"Video Model Keys in Model: {len(video_keys_model)}")

if __name__ == "__main__":
    check_keys()
