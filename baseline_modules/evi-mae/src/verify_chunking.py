
import torch
import sys
import os

# Adjust path to import models
sys.path.append('/home/jaemo/Method/baseline_modules/evi-mae/src')

from models.evi_mae import EVIMAEFT

def test_chunking():
    print("Initializing EVIMAEFT...")
    # Dummy model dicts
    video_model_dict = {
        'img_size': 224, 'patch_size': 16, 'encoder_embed_dim': 384, 
        'encoder_depth': 2, 'encoder_num_heads': 4, 'mlp_ratio': 4., 
        'qkv_bias': True, 'decode_embed_dim': 192, 'decode_num_heads': 4
    }
    imu_model_dict = {
        'patch_size': 16, 'channel_num': 6, 'plot_height': 128, 'target_length': 128,
        'encoder_num_heads': 4, 'encoder_depth': 2, 'encoder_embed_dim': 384,
        'enable_graph': False, 'imu_two_stream': False, 'imu_graph_net': 'gat'
    }
    
    model = EVIMAEFT(label_dim=5, video_model_dict=video_model_dict, imu_model_dict=imu_model_dict)
    model.eval()
    
    # Create dummy data: Batch=1, Sequence=82 (simulating the OOM case)
    B, S = 1, 82
    C_imu = 6
    H_imu, W_imu = 1, 1024 # imu input is 4D (B, C, H, W) -> (B, 6, 1, 1024) approx? 
    # Check forward_embedding input expectations.
    # imu: (B, C, H, W)
    # video: (B, C, T, H, W)
    
    # In forward for sequence:
    # Audio/IMU: (B, S, C, H, W)
    # Video: (B, S, C, T, H, W)
    
    # Let's check dimensions expected by PatchEmbed/PatchEmbed_video
    # IMU PatchEmbed: (H, W) -> (1, 1024) maybe? 
    # forward_embedding calls:
    # a.transpose(2, 3) -> (B, C, W, H)
    # patch_embed_a(a) 
    
    # Let's look at how dataloader loads it.
    # sensor_stack: (S, C, Time, Freq) = (S, 6, 128, 128) based on default target_length?
    # video_stack: (S, C, T, H, W) = (S, 3, 16, 224, 224)
    
    a_dummy = torch.randn(B, S, 6, 128, 128)
    v_dummy = torch.randn(B, S, 3, 16, 224, 224)
    lengths = torch.tensor([82])
    
    print(f"Testing forward with Batch={B}, Sequence={S}, Chunk Size=16...")
    print(f"Changes applied: processing {S} windows in chunks of 16")
    
    with torch.no_grad():
        # mode='multimodal' is standard
        logits = model(a_dummy, v_dummy, mode='multimodal', lengths=lengths, chunk_size=16)
    
    print("Output shape:", logits.shape)
    assert logits.shape == (B, 5), f"Expected (1, 5), got {logits.shape}"
    print("Verification Successful!")

if __name__ == "__main__":
    test_chunking()
