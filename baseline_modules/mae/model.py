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
import torch.nn.functional as F # Contrastive Loss 계산 위해 추가
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# Attention 클래스가 qk_scale을 받지 않으므로 timm의 것을 그대로 사용
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed 

# get_2d_sincos_pos_embed 함수가 다른 파일에 정의되어 있다고 가정
# from .utils import get_2d_sincos_pos_embed 
# ===================================================================
# Placeholder: get_2d_sincos_pos_embed 함수가 없으면 아래 코드를 임시로 사용
# 실제로는 utils.py 등에서 제대로 된 구현을 가져와야 함
import numpy as np
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=False):
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # HxW
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h.shape[0], grid_w.shape[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
# ===================================================================

# random_masking_unstructured 함수도 필요
def random_masking_unstructured(x, mask_ratio):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore, len_keep
# ===================================================================


class PatchEmbed(nn.Module): # timm의 것을 그대로 사용해도 무방할 수 있음
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # grid_size 계산 추가 (원본 timm 코드 참조)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 입력 x의 채널 차원 확인 (디버깅용)
        # print(f"PatchEmbed input shape: {x.shape}") 
        x = self.proj(x).flatten(2).transpose(1, 2)
        print("patch embed shape", x.shape)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 모달리티별 LayerNorm은 CAVMAE 논문 의도가 아니라면 제거 가능
        self.norm1 = norm_layer(dim) 
        # self.norm1_a = norm_layer(dim) # 사용 안 함
        # self.norm1_v = norm_layer(dim) # 사용 안 함
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.norm2_a = norm_layer(dim) # 사용 안 함
        # self.norm2_v = norm_layer(dim) # 사용 안 함
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # [수정 2] forward 인자에서 modality 제거
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    # --- 필요한 import ---
import os
os.environ['TORCH_HOME'] = './pretrained_models' # 이 줄은 환경에 따라 필요 없을 수 있음
import random
import torch
import torch.nn as nn
import torch.nn.functional as F # 추가
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# Attention 클래스가 qk_scale을 받지 않으므로 timm의 것을 그대로 사용
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

# --- Placeholder 함수 (실제 구현으로 대체 필요) ---
import numpy as np
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=False):
    # (이전 턴의 코드 붙여넣기)
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h); grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h.shape[0], grid_w.shape[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token: pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1); return emb
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0; omega = np.arange(embed_dim // 2, dtype=np.float32); omega /= embed_dim / 2.; omega = 1. / 10000**omega
    pos = pos.reshape(-1); out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out); emb_cos = np.cos(out); emb = np.concatenate([emb_sin, emb_cos], axis=1); return emb
def random_masking_unstructured(x, mask_ratio):
    N, L, D = x.shape; len_keep = int(L * (1 - mask_ratio)); noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1); ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device); mask[:, :len_keep] = 0; mask = torch.gather(mask, dim=1, index=ids_restore)
    # len_keep 반환 추가
    return x_masked, mask, ids_restore, len_keep
# 헬퍼 함수
def to_tuple(x): return x if isinstance(x, tuple) else (x, x)
# --- Placeholder 끝 ---

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x): # modality 인자 제거
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class CAVMAE(nn.Module):
    def __init__(self,
                 sensor_in_chans=37, sensor_seq_len=128, sensor_patch_size=(4, 16),
                 img_size=224, video_patch_size=16, video_in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        print('An adapted CAV-MAE Model for Sensor and Video (Padding-based)')

        # --- 패딩 계산 로직 추가 ---
        self.sensor_patch_size = to_tuple(sensor_patch_size)
        ph, pw = self.sensor_patch_size
        H_orig, W_orig = sensor_in_chans, sensor_seq_len
        H_padded, W_padded = H_orig, W_orig
        self.pad_h, self.pad_w = 0, 0
        if H_orig % ph != 0: 
            self.pad_h = ph - (H_orig % ph)
            H_padded += self.pad_h
        if W_orig % pw != 0:
            self.pad_w = pw - (W_orig % pw)
            W_padded += self.pad_w
        
        print(f"Sensor padding: H={H_orig}->{H_padded}, W={W_orig}->{W_padded}")
        # --- 패딩 계산 끝 ---

        # PatchEmbed 초기화 시 패딩된 크기 사용
        self.patch_embed_s = PatchEmbed(img_size=(H_padded, W_padded), patch_size=self.sensor_patch_size, in_chans=1, embed_dim=embed_dim) 
        self.patch_embed_v = PatchEmbed(img_size=img_size, patch_size=video_patch_size, in_chans=video_in_chans, embed_dim=embed_dim)

        self.modality_s = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Embedding (패딩된 크기 기준 num_patches 사용)
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.patch_embed_s.num_patches, embed_dim), requires_grad=False)
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=False)

        # 이하 __init__ 동일 ...
        self.blocks_s = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(12 - modality_specific_depth)])
        self.norm_s, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_s = nn.Parameter(torch.zeros(1, self.patch_embed_s.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_s = nn.Linear(decoder_embed_dim, self.sensor_patch_size[0] * self.sensor_patch_size[1] * 1, bias=True) # in_chans=1
        self.video_patch_size_tuple = to_tuple(video_patch_size) # video_patch_size가 int일 수 있으므로 튜플 변환
        self.decoder_pred_v = nn.Linear(decoder_embed_dim, self.video_patch_size_tuple[0] ** 2 * video_in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # 위치 임베딩 생성 시 patch_embed에서 계산된 grid_size 사용
        grid_h_s, grid_w_s = self.patch_embed_s.grid_size # 패딩된 크기 기준 grid_size
        pos_embed_s = get_2d_sincos_pos_embed(
            self.pos_embed_s.shape[-1], grid_h_s, grid_w_s, cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

        grid_h_v, grid_w_v = self.patch_embed_v.grid_size
        pos_embed_v = get_2d_sincos_pos_embed(
            self.pos_embed_v.shape[-1], grid_h_v, grid_w_v, cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        decoder_pos_embed_s = get_2d_sincos_pos_embed(
            self.decoder_pos_embed_s.shape[-1], grid_h_s, grid_w_s, cls_token=False)
        self.decoder_pos_embed_s.data.copy_(torch.from_numpy(decoder_pos_embed_s).float().unsqueeze(0))

        decoder_pos_embed_v = get_2d_sincos_pos_embed(
            self.decoder_pos_embed_v.shape[-1], grid_h_v, grid_w_v, cls_token=False)
        self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))

        # 이하 초기화 동일...
        w = self.patch_embed_s.proj.weight.data; torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data; torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.modality_s, std=.02); torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    # patchify 함수 수정: 패딩된 크기 기준으로 작동하도록 수정 불필요 (이미 패딩된 데이터가 들어옴)
    def patchify(self, data, patch_size, in_chans):
        if data.shape[1] != in_chans: data = data.unsqueeze(1)
        B, C, H, W = data.shape
        ph, pw = to_tuple(patch_size)
        # 패딩된 데이터가 들어오므로 assert는 통과해야 함
        assert C == in_chans and H % ph == 0 and W % pw == 0, f"Data shape {data.shape} not divisible by patch size {patch_size}"
        h_patches, w_patches = H // ph, W // pw
        x = data.reshape(B, C, h_patches, ph, w_patches, pw)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h_patches * w_patches, ph * pw * C)
        return x

    def forward_encoder(self, sensor, video, mask_ratio_s, mask_ratio_v):
        # sensor는 이미 패딩된 (B, 1, H_padded, W_padded) 형태
        
        # PatchEmbed + PosEmbed + ModalityEmbed (크기 일치)
        s = self.patch_embed_s(sensor) + self.pos_embed_s + self.modality_s # 이제 크기 맞음!
        
        # 비디오 처리 (이전 수정 반영)
        B, T, C, H, W = video.shape 
        video_reshaped = video.reshape(B * T, C, H, W)
        v_patches = self.patch_embed_v(video_reshaped) 
        num_patches_v_per_frame = self.patch_embed_v.num_patches 
        v = v_patches.reshape(B, T * num_patches_v_per_frame, -1) 
        pos_embed_v_repeated = self.pos_embed_v.repeat(1, T, 1) 
        v = v + pos_embed_v_repeated + self.modality_v 

        # 마스킹 (len_keep 반환 추가됨 가정)
        s, mask_s, ids_restore_s, len_s_keep = random_masking_unstructured(s, mask_ratio_s)
        v, mask_v, ids_restore_v, len_v_keep = random_masking_unstructured(v, mask_ratio_v)
        
        # 블록 통과 (modality 인자 없음)
        for blk in self.blocks_s: s = blk(s) 
        for blk in self.blocks_v: v = blk(v)

        x = torch.cat((s, v), dim=1)
        for blk in self.blocks_u: x = blk(x)
        x = self.norm(x)

        # Contrastive Rep. (이전과 동일, 마스킹된 상태 유의)
        latent_c_s = self.norm_s(s)
        latent_c_v = self.norm_v(v)

        return x, mask_s, ids_restore_s, mask_v, ids_restore_v, latent_c_s, latent_c_v, len_s_keep, len_v_keep

    def forward_decoder(self, x, ids_restore_s, ids_restore_v, len_s_keep, len_v_keep):
        x = self.decoder_embed(x)
        
        # 총 패치 수 계산 (패딩된 크기 기준)
        num_s_patches_total = self.patch_embed_s.num_patches # 예: 24
        T = ids_restore_v.shape[1] // self.patch_embed_v.num_patches 
        num_v_patches_total = T * self.patch_embed_v.num_patches # 예: 3136
        
        # 보이는 패치 분리
        x_s = x[:, :len_s_keep]
        x_v = x[:, len_s_keep:] # 수정됨

        # 마스크 토큰 추가 및 Gather (이전 수정 반영)
        mask_tokens_s = self.mask_token.repeat(x.shape[0], num_s_patches_total - len_s_keep, 1)
        s_ = torch.cat([x_s, mask_tokens_s], dim=1)
        s_ = torch.gather(s_, dim=1, index=ids_restore_s.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        mask_tokens_v = self.mask_token.repeat(x.shape[0], num_v_patches_total - len_v_keep, 1)
        v_ = torch.cat([x_v, mask_tokens_v], dim=1)
        v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # 디코더 위치 임베딩 추가 (패딩된 크기 기준)
        decoder_pos_embed_v_repeated = self.decoder_pos_embed_v.repeat(1, T, 1)
        s_ = s_ + self.decoder_pos_embed_s
        v_ = v_ + decoder_pos_embed_v_repeated 

        x = torch.cat([s_, v_], dim=1)

        for blk in self.decoder_blocks: x = blk(x)
        x = self.decoder_norm(x)

        pred_s = self.decoder_pred_s(x[:, :num_s_patches_total])
        pred_v = self.decoder_pred_v(x[:, num_s_patches_total:])
        
        return pred_s, pred_v
    
    # forward_contrastive (이전과 동일)
    def forward_contrastive(self, s_rep, v_rep):
        s_rep_agg = s_rep.mean(dim=1); v_rep_agg = v_rep.mean(dim=1)
        s_rep_norm = F.normalize(s_rep_agg, dim=-1); v_rep_norm = F.normalize(v_rep_agg, dim=-1)
        total = torch.mm(s_rep_norm, v_rep_norm.t()) / 0.05
        nce = -torch.mean(torch.diag(F.log_softmax(total, dim=1)))
        c_acc = (torch.argmax(total, dim=1) == torch.arange(len(total), device=s_rep.device)).float().mean()
        return nce, c_acc

    # forward_mae_loss (패딩 로직 제거, 슬라이싱 유지)
    def forward_mae_loss(self, input_data, pred, mask, patch_size, in_chans):
        
        # --- ★★★ 비디오/센서 분기 처리 추가 ★★★ ---
        if in_chans == 3: # 비디오 데이터인 경우
            B, T, C, H, W = input_data.shape # (B, T, C, H, W)
            # (B, T, C, H, W) -> (B * T, C, H, W)로 변경
            input_data_reshaped = input_data.reshape(B * T, C, H, W)
            
            # 4D 텐서로 patchify 호출
            target_patches = self.patchify(input_data_reshaped, patch_size, in_chans) 
            # target_patches shape: (B*T, N_v, P*P*C), N_v=프레임당 패치 수
            
            # pred_v shape: (B, T*N_v, P*P*C) 와 맞추기 위해 reshape
            num_patches_per_frame = target_patches.shape[1]
            target = target_patches.reshape(B, T * num_patches_per_frame, -1)
            
        else: # 센서 데이터인 경우 (in_chans == 1)
            # 이전 패딩 로직은 forward 함수로 이동했으므로 여기선 제거
            # input_data는 이미 패딩된 (B, 1, H_padded, W_padded) 형태
            target = self.patchify(input_data, patch_size, in_chans) # 패딩된 데이터 기준 패치 생성

            # pred_s shape: (B, N_s_padded, P*P*C)
            # target shape: (B, N_s_padded, P*P*C)
            # 패딩으로 인해 pred와 target의 패치 수가 다를 수 있으므로 슬라이싱 필요
            num_pred_patches = pred.shape[1] 
            target = target[:, :num_pred_patches, :] # target 길이를 pred 길이에 맞춤
            
        # --- 분기 처리 끝 ---

        # --- 이하 MAE Loss 계산 동일 ---
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # [N, L]

        # 마스크 크기도 pred 크기에 맞춰야 함
        if mask.shape[1] != pred.shape[1]:
             mask = mask[:, :pred.shape[1]] 

        loss = (loss * mask).sum() / mask.sum() 
        return loss

    # forward (패딩 로직 이동)
    def forward(self, sensor, video, mask_ratio_s, mask_ratio_v, mae_loss_weight, contrast_loss_weight):
        # --- 패딩 로직 ---
        if len(sensor.shape) == 3: sensor = sensor.unsqueeze(1)
        B, C_in, H, W = sensor.shape
        ph, pw = self.sensor_patch_size # __init__에서 to_tuple 완료
        pad_h, pad_w = 0, 0
        if H % ph != 0: pad_h = ph - (H % ph)
        if W % pw != 0: pad_w = pw - (W % pw)
        if pad_h > 0 or pad_w > 0:
            padding = (0, pad_w, 0, pad_h)
            sensor_padded = F.pad(sensor, padding, mode='constant', value=0) 
        else:
            sensor_padded = sensor
        # --- 패딩 끝 ---

        # 패딩된 센서 데이터 사용
        latent, mask_s, ids_restore_s, mask_v, ids_restore_v, latent_c_s, latent_c_v, len_s_keep, len_v_keep = \
            self.forward_encoder(sensor_padded, video, mask_ratio_s, mask_ratio_v)
        
        pred_s, pred_v = self.forward_decoder(latent, ids_restore_s, ids_restore_v, len_s_keep, len_v_keep)
        
        # MAE Loss 계산 시 패딩된 센서 사용
        loss_mae_s = self.forward_mae_loss(sensor_padded, pred_s, mask_s, self.sensor_patch_size, 1) 
        loss_mae_v = self.forward_mae_loss(video, pred_v, mask_v, self.video_patch_size_tuple, 3) # __init__에서 튜플로 변환됨
        loss_mae = mae_loss_weight * (loss_mae_s + loss_mae_v)
        
        loss_c, c_acc = self.forward_contrastive(latent_c_s, latent_c_v)
        loss_c = contrast_loss_weight * loss_c
        
        loss = loss_mae + loss_c
        
        return loss, loss_mae, loss_mae_s, loss_mae_v, loss_c, c_acc
    # CAVMAE 클래스 내부

    def forward_sensor_only(self, sensor):
        # 1. 패딩 (forward 함수와 동일하게 적용)
        if len(sensor.shape) == 3: sensor = sensor.unsqueeze(1)
        B, C_in, H, W = sensor.shape
        ph, pw = self.sensor_patch_size 
        pad_h, pad_w = 0, 0
        if H % ph != 0: pad_h = ph - (H % ph)
        if W % pw != 0: pad_w = pw - (W % pw)
        if pad_h > 0 or pad_w > 0:
            padding = (0, pad_w, 0, pad_h)
            sensor_padded = F.pad(sensor, padding, mode='constant', value=0) 
        else:
            sensor_padded = sensor

        # 2. Patch Embedding + Positional Embedding + Modality Token
        s_patches = self.patch_embed_s(sensor_padded) 
        # 위치 임베딩 크기 조절 (만약 interpolate 방식을 쓴다면)
        # pos_embed_s_resized = self.interpolate_pos_encoding(s_patches, self.pos_embed_s, self.patch_embed_s)
        # s = s_patches + pos_embed_s_resized + self.modality_s
        # 또는 초기화 시 패딩된 크기 기준으로 만들었다면 바로 더하기
        s = s_patches + self.pos_embed_s
        s += self.modality_s 

        # 3. 센서 전용 블록 통과
        for blk in self.blocks_s:
            s = blk(s)

        # --- 여기서 선택 ---
        # 옵션 1: 통합 블록(blocks_u) 통과시키기 (EVIMAEFT의 audioonly 방식)
        # for blk in self.blocks_u:
        #     s = blk(s) # modality 인자 없이? 또는 's'? 확인 필요
        # s = self.norm_s(s) # 센서 전용 norm 사용? 아니면 self.norm?

        # 옵션 2: 통합 블록 건너뛰기 (센서 인코더만의 특징 확인)
        s = self.norm_s(s) # 센서 전용 norm만 사용
        # --- 선택 끝 ---

        # 4. 최종 특징 벡터 생성 (예: 평균 풀링)
        x = s.mean(dim=1) 
        return x
