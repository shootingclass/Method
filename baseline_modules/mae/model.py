# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block

from utils import get_2d_sincos_pos_embed

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x
    
class CAVMAE(nn.Module):
    """
    CAV-MAE 모델을 유연한 센서 채널과 비디오 데이터에 맞게 수정한 버전입니다.
    EVI-MAE의 그래프나 신체 부위 특정 로직은 사용하지 않습니다.
    """
    def __init__(self,
                 sensor_in_chans=36, sensor_seq_len=128, sensor_patch_size=(4, 16),
                 img_size=224, video_patch_size=16, video_in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        print('An adapted CAV-MAE Model for Sensor and Video with Full Decoder')

        # --- 1. 인코더 부분 ---
        # 센서를 (채널 수, 시간 길이) 모양의 2D 이미지로 취급
        self.patch_embed_s = PatchEmbed(img_size=(sensor_in_chans, sensor_seq_len), patch_size=sensor_patch_size, in_chans=1, embed_dim=embed_dim)
        self.patch_embed_v = PatchEmbed(img_size=img_size, patch_size=video_patch_size, in_chans=video_in_chans, embed_dim=embed_dim)

        self.modality_s = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.patch_embed_s.num_patches, embed_dim), requires_grad=False)
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=False)

        self.blocks_s = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(12 - modality_specific_depth)])

        self.norm_s, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)

        # --- 2. 디코더 부분 ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed_s = nn.Parameter(torch.zeros(1, self.patch_embed_s.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_pred_s = nn.Linear(decoder_embed_dim, sensor_patch_size[0] * sensor_patch_size[1] * 1, bias=True)
        self.decoder_pred_v = nn.Linear(decoder_embed_dim, video_patch_size ** 2 * video_in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # 2D sin-cos 위치 임베딩 초기화
        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], self.patch_embed_s.grid_size[0], self.patch_embed_s.grid_size[1])
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))
        decoder_pos_embed_s = get_2d_sincos_pos_embed(self.decoder_pos_embed_s.shape[-1], self.patch_embed_s.grid_size[0], self.patch_embed_s.grid_size[1])
        self.decoder_pos_embed_s.data.copy_(torch.from_numpy(decoder_pos_embed_s).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], self.patch_embed_v.grid_size[0], self.patch_embed_v.grid_size[1])
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))
        decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], self.patch_embed_v.grid_size[0], self.patch_embed_v.grid_size[1])
        self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))

        # 기타 가중치 초기화
        torch.nn.init.normal_(self.modality_s, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, data, patch_size, in_chans):
        """데이터를 패치 단위로 분할합니다."""
        B, C, H, W = data.shape
        ph, pw = patch_size
        assert C == in_chans and H % ph == 0 and W % pw == 0
        h_patches, w_patches = H // ph, W // pw
        
        x = data.reshape(B, C, h_patches, ph, w_patches, pw)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h_patches * w_patches, ph * pw * C)
        return x

    def forward_encoder(self, sensor, video, mask_ratio_s, mask_ratio_v):
        # 센서를 1채널 2D 이미지로 변환 (B, C, T) -> (B, 1, C, T)
        sensor = sensor.unsqueeze(1)
        
        s = self.patch_embed_s(sensor) + self.pos_embed_s + self.modality_s
        v = self.patch_embed_v(video) + self.pos_embed_v + self.modality_v

        s, mask_s, ids_restore_s = random_masking_unstructured(s, mask_ratio_s)
        v, mask_v, ids_restore_v = random_masking_unstructured(v, mask_ratio_v)
        
        for blk in self.blocks_s:
            s = blk(s, 'a')
        for blk in self.blocks_v:
            v = blk(v, 'v')

        x = torch.cat((s, v), dim=1)
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm(x)

        # Contrastive Loss를 위한 표현(representation)
        # 마스킹되지 않은 패치들만 사용
        latent_c_s = self.norm_s(s)
        latent_c_v = self.norm_v(v)

        return x, mask_s, ids_restore_s, mask_v, ids_restore_v, latent_c_s, latent_c_v

    def forward_decoder(self, x, ids_restore_s, ids_restore_v):
        x = self.decoder_embed(x)
        
        num_s_patches_kept = x.shape[1] - self.patch_embed_v.num_patches
        num_v_patches_kept = self.patch_embed_v.num_patches
        
        # 센서와 비디오의 보이는(visible) 패치를 분리
        x_s = x[:, :num_s_patches_kept]
        x_v = x[:, num_s_patches_kept:]

        # 마스크 토큰 추가
        mask_tokens_s = self.mask_token.repeat(x.shape[0], self.patch_embed_s.num_patches - num_s_patches_kept, 1)
        s_ = torch.cat([x_s, mask_tokens_s], dim=1)
        s_ = torch.gather(s_, dim=1, index=ids_restore_s.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        mask_tokens_v = self.mask_token.repeat(x.shape[0], self.patch_embed_v.num_patches - num_v_patches_kept, 1)
        v_ = torch.cat([x_v, mask_tokens_v], dim=1)
        v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # 위치 임베딩 추가
        s_ = s_ + self.decoder_pos_embed_s
        v_ = v_ + self.decoder_pos_embed_v

        x = torch.cat([s_, v_], dim=1)

        # 디코더 블록 통과
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 최종 예측
        pred_s = self.decoder_pred_s(x[:, :self.patch_embed_s.num_patches])
        pred_v = self.decoder_pred_v(x[:, self.patch_embed_s.num_patches:])
        
        return pred_s, pred_v
    
    def forward_contrastive(self, s_rep, v_rep):
        s_rep_agg = s_rep.mean(dim=1)
        v_rep_agg = v_rep.mean(dim=1)
        
        s_rep_norm = F.normalize(s_rep_agg, dim=-1)
        v_rep_norm = F.normalize(v_rep_agg, dim=-1)
        
        total = torch.mm(s_rep_norm, v_rep_norm.t()) / 0.05
        nce = -torch.mean(torch.diag(F.log_softmax(total, dim=1)))
        c_acc = (torch.argmax(total, dim=1) == torch.arange(len(total), device=s_rep.device)).float().mean()
        return nce, c_acc

    def forward_mae_loss(self, input_data, pred, mask, patch_size, in_chans=1):
        target = self.patchify(input_data, patch_size, in_chans)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # [N, L]
        loss = (loss * mask).sum() / mask.sum() # 마스킹된 패치에 대해서만 평균
        return loss

    def forward(self, sensor, video, mask_ratio_s, mask_ratio_v, mae_loss_weight, contrast_loss_weight):
        # 센서를 1채널 2D 이미지로 변환
        sensor_img = sensor.unsqueeze(1)

        latent, mask_s, ids_restore_s, mask_v, ids_restore_v, latent_c_s, latent_c_v = \
            self.forward_encoder(sensor, video, mask_ratio_s, mask_ratio_v)
        
        # 디코더를 통해 마스킹된 패치 예측
        pred_s, pred_v = self.forward_decoder(latent, ids_restore_s, ids_restore_v)
        
        # MAE Loss 계산
        loss_mae_s = self.forward_mae_loss(sensor_img, pred_s, mask_s, self.patch_embed_s.patch_size, 1)
        loss_mae_v = self.forward_mae_loss(video, pred_v, mask_v, self.patch_embed_v.patch_size, 3)
        loss_mae = mae_loss_weight * (loss_mae_s + loss_mae_v)
        
        # Contrastive Loss 계산
        loss_c, c_acc = self.forward_contrastive(latent_c_s, latent_c_v)
        loss_c = contrast_loss_weight * loss_c
        
        # 최종 Loss
        loss = loss_mae + loss_c
        
        return loss, loss_mae, loss_mae_s, loss_mae_v, loss_c, c_acc

