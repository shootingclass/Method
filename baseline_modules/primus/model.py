
# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
import pytorch_lightning as pl
import torch
from typing import List, Optional
import numpy as np
import clip
import json
from PIL import Image
from torchvision.transforms import Normalize
from transformers import CLIPVisionModelWithProjection
from matplotlib import cm
import torch.nn.functional as F

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_type="max", embedding_size=32):
        super().__init__()
        if pool_type == "max":
            pool_fn = torch.nn.MaxPool1d(kernel_size=3)
        elif pool_type == "adaptive":
            pool_fn = torch.nn.AdaptiveAvgPool1d(output_size=embedding_size)
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2,
                bias=False,
            ),
            pool_fn,
        )

    def forward(self, batch):
        return self.net(batch)


class MW2StackRNNPooling(pl.LightningModule):
    def __init__(self, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.name = MW2StackRNNPooling
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(2, 6),
            Block(6, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5, pool_type="adaptive", embedding_size=32),
            torch.nn.GroupNorm(4, input_dim),
            torch.nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )

    def forward(self, batch):
        # return the last hidden state
        return self.net(batch)[1][0]

class MW2StackRNNPoolingMultihead(pl.LightningModule):
    def __init__(self, num_sensors=37, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.name = MW2StackRNNPooling
        self.backbone = torch.nn.Sequential(
            torch.nn.GroupNorm(1, num_sensors),
            Block(num_sensors, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5, pool_type="adaptive", embedding_size=32),
            torch.nn.GroupNorm(4, input_dim),
            torch.nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )
        self.ssl_head = torch.nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = torch.nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch):
        emb = self.backbone(batch)[1][0] # Last hidden state
        ssl_out = self.ssl_head(emb)
        mmcl_out = self.mmcl_head(emb)
        out = {"ssl": ssl_out, "mmcl": mmcl_out, "emb": emb}
        return out
    
# class Clip4CLIPModel(pl.LightningModule):

#     def __init__(self, freeze):
#         super(Clip4CLIPModel, self).__init__()
#         print("Loading clip4clip model ...")

#         self.flag_freeze = freeze
#         from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig

#         config = CLIPVisionModelWithProjection.from_pretrained(
#             "openai/clip-vit-base-patch32"
#         ).config

#         config.output_hidden_states = True  # 🔥 핵심
#         config.return_dict = True           # 안전하게

#         self.video_model = CLIPVisionModelWithProjection.from_pretrained(
#             "openai/clip-vit-base-patch32",
#             config=config
#         )
#         # self.video_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

#         self.video_model.eval()

#         if self.flag_freeze:
#             self.eval()
#             self.freeze()

class Clip4CLIPModel(pl.LightningModule):

    def __init__(self, freeze, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super(Clip4CLIPModel, self).__init__()
        print("Loading clip4clip model ...")

        self.flag_freeze = freeze
        self.video_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        if self.flag_freeze:
            # 완전 freeze (기존 동작)
            self.video_model.eval()
            self.eval()
            self.freeze()
        else:
            # base 가중치 freeze + LoRA 어댑터만 학습
            for param in self.video_model.parameters():
                param.requires_grad = False

            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.video_model = get_peft_model(self.video_model, lora_config)
            self.video_model.print_trainable_parameters()

    def get_video_embeddings(self, video, device: Optional[str] = None):
        """
        video: [B, T, 3, H, W] 또는 [B, T, H, W, 3]
        return:
          video_features: [B, D]
          overlay_list:   길이 B*T 의 numpy uint8 이미지 리스트
        """

        # precomputed features인 경우
        if len(video.shape) == 2:
            return video, None

        # --- 0) 입력 shape 정리 ---
        if video.shape[2] == 3:
            # [B, T, 3, H, W]
            B, T, C, H, W = video.shape
            video_btchw = video
        else:
            # [B, T, H, W, 3]
            B, T, H, W, C = video.shape
            video_btchw = video.permute(0, 1, 4, 2, 3)  # → [B,T,3,H,W]

        video_orig = video_btchw  # overlay용 원본 보관
        video_flat = video_btchw.reshape(B * T, 3, H, W)  # CLIP 입력

        # --- 1) CLIP forward ---
        visual_output_raw = self.video_model(video_flat)

        # 🔑 여기! last_hidden_state는 vision_model_output 안에 있음
        token_feats = visual_output_raw.last_hidden_state
        # token_feats: [B*T, 50, 768]  (CLS + 49 patch)

        # --- 2) patch token → spatial map ---
        patch_tokens = token_feats[:, 1:, :]           # [B*T, 49, 768]
        num_patches = patch_tokens.shape[1]            # 49
        S = int(num_patches ** 0.5)                    # 7
        spatial_map = patch_tokens.reshape(B * T, S, S, patch_tokens.shape[-1])  # [B*T, 7,7,768]

        # --- 3) frame별 overlay 생성 ---
        overlay_list = []

        for idx in range(B * T):
            b = idx // T
            t = idx % T

            # feature map → heatmap
            feat = spatial_map[idx]                 # [7,7,768]
            heat = feat.mean(-1).detach().cpu().numpy()      # [7,7]

            hmin, hmax = heat.min(), heat.max()
            if hmax == hmin:
                heat_norm = np.zeros_like(heat)
            else:
                heat_norm = (heat - hmin) / (hmax - hmin + 1e-8)

            cmap = cm.get_cmap("jet")
            heat_rgb = cmap(heat_norm)[:, :, :3]    # [7,7,3], 0~1

            heat_t = torch.from_numpy(heat_rgb).permute(2, 0, 1)[None].float()
            heat_up = F.interpolate(
                heat_t, size=(H, W), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0).numpy()          # [H,W,3]

            # 원본 프레임
            frame = video_orig[b, t].detach().cpu()  # [3,H,W]
            if frame.min() < 0:  # -1~1 → 0~1
                frame = (frame + 1) / 2.0
            frame_np = frame.permute(1, 2, 0).numpy()  # [H,W,3]

            # overlay
            alpha = 0.45
            overlay = (1 - alpha) * frame_np + alpha * heat_up
            overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

            overlay_list.append(overlay_uint8)

        # --- 4) frame 평균 CLIP embedding ---
        video_features = visual_output_raw.image_embeds  # [B*T, D]
        video_features = video_features.reshape(B, T, -1)
        video_features = video_features.mean(dim=1)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features, overlay_list
