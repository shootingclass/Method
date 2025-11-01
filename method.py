import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt
import imageio
import math
import torch.distributed as dist
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from method_utils import log_video_recon_grid, log_video_recon_gif, match_target_to_recon, to_vis_motion, CovarianceAlignmentLoss, overlay_motion_heatmap, log_optical_flow_overlay_to_wandb
import copy

# --- 사용자 정의 모듈 임포트 ---
from model import SensorModel, VisionModel, ClusteringModel

####################################################################



class MethodLightningModule(pl.LightningModule):

    def __init__(self, args, datamodule=None):
        super().__init__()
        self.save_hyperparameters(args)
        # 1. 모델 구성 요소 초기화

        self.video_model = VisionModel(latent_dim=self.hparams.embedding_dim)
        self.sensor_model = SensorModel(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.sensor_motion_model = SensorModel(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.clustering_model = ClusteringModel(
            encoder=self.sensor_model,
            embedding_dim=self.hparams.embedding_dim,
            num_sensors=self.hparams.num_sensors,
            num_clusters=self.hparams.num_classes,
            datamodule=datamodule,
            top_k=self.hparams.top_k,
            prototype_cache_dir=os.path.join(self.hparams.cache_dir, "prototypes"),
            dataset_name=self.hparams.dataset_name,
            min_cluster_size=self.hparams.min_cluster_size
        )

        self.appearance_classifier = nn.Linear(256, self.hparams.num_classes) 
    
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.success_labels=[0 for i in range(self.hparams.num_classes)]
        self.fail_labels=[0 for i in range(self.hparams.num_classes)]
        self.video_classifier = nn.Linear(self.hparams.embedding_dim, self.hparams.num_classes)
        print(self.global_rank, "Model initialized.")
        self.mutual_information_loss_fn = CovarianceAlignmentLoss()
        self.epoch = 0        
        # =================================================================
        
        self.momentum_video_model = copy.deepcopy(self.video_model)
        for param in self.momentum_video_model.parameters():
            param.requires_grad = False
        # =================================================================

                                                             
        # =================================================================
        # [1단계-B] 모멘텀 센서 모션 모델 추가
        # =================================================================
        self.momentum_sensor_motion_model = copy.deepcopy(self.sensor_motion_model)
        for param in self.momentum_sensor_motion_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoders(self):
        """
        모멘텀 인코더 파라미터를 온라인 인코더 파라미터로
        Exponential Moving Average (EMA) 업데이트를 수행합니다.
        """
        # __init__에서 저장한 하이퍼파라미터 (e.g., args에 --momentum_m 0.999 포함)
        m = self.hparams.momentum_m 
        
        # 1. 비디오 모델 파라미터 업데이트
        for param_q, param_k in zip(self.video_model.parameters(), 
                                self.momentum_video_model.parameters()):
            # param_k = param_k * m + param_q * (1. - m)
            param_k.data = param_k.data * m + param_q.data * (1. - m)
            
        # 2. 센서 모션 모델 파라미터 업데이트
        for param_q, param_k in zip(self.sensor_motion_model.parameters(), 
                                self.momentum_sensor_motion_model.parameters()):
            # param_k = param_k * m + param_q * (1. - m)
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def enable_stage2(self):
        """appearance는 고정, motion만 학습"""
        # 1) appearance 관련 모듈 동결 (scene/object/fuse/backbone 등)
        for n, p in self.video_model.named_parameters():
            if any(key in n for key in ["shared_encoder", "scene", "object", "fuse", "appearance", "video_classifier"]):
                p.requires_grad = False

        # 2) 모션 브랜치만 학습
        for n, p in self.video_model.named_parameters():
            if "motion_branch" in n:
                p.requires_grad = True

        self.stage2 = True  # flag

    def _calculate_alignment_loss(self, v_app: torch.Tensor, sensor: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07, symmetric: bool = True):
        """
        Cross-modal supervised contrastive (InfoNCE) between appearance and sensor embeddings.
        v_app:   [N, D]
        sensor:  [N, D]
        labels:  [N]  (pseudo labels; same class = positive)
        """
        # L2 normalize
        v_app   = F.normalize(v_app,   dim=1)
        sensor  = F.normalize(sensor,  dim=1)

        # [N, N] similarity / temperature
        logits_as = (v_app @ sensor.T) / temperature   # anchor = appearance, samples = sensor
        logits_sa = (sensor @ v_app.T) / temperature   # anchor = sensor,    samples = appearance

        # positive mask: same class → 1, else 0
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(v_app.device)
        # 각 anchor마다 최소 1개 양성 보장이 안 될 수도 있으므로 분모 보호
        pos_cnt = pos_mask.sum(dim=1).clamp(min=1.0)

        # row-wise log-softmax
        logprob_as = logits_as.log_softmax(dim=1)  # appearance→sensor
        logprob_sa = logits_sa.log_softmax(dim=1)  # sensor→appearance

        # supervised contrastive: 양성들 평균 negative log-likelihood
        loss_as = -(logprob_as * pos_mask).sum(dim=1) / pos_cnt
        if symmetric:
            loss_sa = -(logprob_sa * pos_mask).sum(dim=1) / pos_cnt
            loss = 0.5 * (loss_as.mean() + loss_sa.mean())
        else:
            loss = loss_as.mean()

        return loss


    # 에포크 시작 시 clustering_model 상태 업데이트
    def on_train_epoch_start(self):

        # 첫 에폭에서 메모리 뱅크 초기화 (중요!)
        if self.epoch == 0:
            print("Initializing memory bank at epoch 0...")
            self.clustering_model.init_prototypes_with_data(self.device, self.hparams.num_classes)
            if self.hparams.threshold_epoch == -1:
                ckpt_path = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/caches/clustering_model_stage1_epoch=5.pt"
                state = torch.load(ckpt_path, map_location="cpu")
                self.clustering_model.load_state_dict(state["cluster_model"])
                self.clustering_model.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_model.clustering_manager.centroids = state["centroids"].to(self.device)

                # ⚠️ 중요: buffer까지 GPU로 강제 이동
                self.clustering_model.to(self.device)

                print(f"[✔] Loaded pretrained clustering model from {ckpt_path}")
                           # ✅ Stage2-only 모드 (캐시 로드)
            if self.hparams.threshold_epoch == 0 and self.hparams.video_classifier_epoch == 0:
                ckpt_path = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/caches/vision_model_stage1_epoch=14.pt"
                print(f"[Stage2-Only Mode] Loading pretrained clustering model from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                self.clustering_model.load_state_dict(state["cluster_model"])
                self.clustering_model.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_model.clustering_manager.centroids = state["centroids"].to(self.device)
                self.clustering_model.to(self.device)

                vision_state_dict = state.get("vision_model", state)

                # ✅ 현재 모델 state_dict 불러오기
                current_state = self.video_model.state_dict()

                # ✅ 키 일치하고, shape도 일치하는 것만 남기기
                filtered_state = {k: v for k, v in vision_state_dict.items()
                                if k in current_state and v.shape == current_state[k].shape}

                # ✅ 로드
                missing, unexpected = self.video_model.load_state_dict(filtered_state, strict=False)

                print(f" → loaded with missing: {missing}, unexpected: {unexpected}")
                self.video_model.to(self.device)

                # ⚡ 바로 Stage2 시작 모드 플래그
                self.stage2_only = True
                self.enable_stage2()
            else:
                self.stage2_only = False


        self.training_steps_outputs = []  # 에포크 동안의 출력 저장용

        if self.epoch == self.hparams.threshold_epoch - 1 and self.global_rank == 0:
            ckpt_path = os.path.join("/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/caches", f"clustering_model_stage1_epoch={self.hparams.threshold_epoch}.pt")
            torch.save({
                "cluster_model": self.clustering_model.state_dict(),
                "memory": self.clustering_model.clustering_manager.feature_bank,
                "centroids": self.clustering_model.clustering_manager.centroids,
            }, ckpt_path)

            print(f"[✔] Saved clustering stage-1 weights → {ckpt_path}")
        
        if self.epoch == self.hparams.threshold_epoch + self.hparams.video_classifier_epoch + self.hparams.bad_correction_epoch:
            stage1_ckpt_path = os.path.join(self.hparams.cache_dir, f"vision_model_stage1_epoch={self.epoch}.pt")
            torch.save({
                "cluster_model": self.clustering_model.state_dict(),
                "memory": self.clustering_model.clustering_manager.feature_bank,
                "centroids": self.clustering_model.clustering_manager.centroids,
                "label_bank": self.clustering_model.clustering_manager.label_bank,
                "vision_model": self.video_model.state_dict()},
            stage1_ckpt_path)
            print(f"[✔] Saved VisionModel weights → {stage1_ckpt_path}")


    def training_step(self, batch, batch_idx):
  
        # 0. 데이터 준비 (Lightning이 자동으로 device로 옮겨줍니다)
        videos, sensors, labels, sample_ids, flows = batch
        idx, sample_id = sample_ids

        # 실제 배치 크기를 텐서에서 직접 가져옵니다.
        current_batch_size = videos.size(0)


        # --- 1. 클러스터링 단계 ---
        
        # 데이터 증강 적용 (선택적)
        if self.clustering_model.epoch > 1:  # 첫 에폭은 원본 데이터로 클러스터링
            imu_data_aug = self.clustering_model.augment_imu_data(sensors)
        else:
            imu_data_aug = sensors
        
        # 모델 순전파
        scores, features, _ = self.clustering_model(imu_data_aug, return_features=True, labels=labels, idx=idx)
        
        # 메모리 뱅크에서 저장된 pseudo label 가져오기
        stored_pseudo_labels = self.clustering_model.get_pseudo_labels(idx)

        scores_original = scores.clone()
        # scores = F.normalize(scores, dim=1)

        # --- [수정된 로직 시작: 클러스터별 75% 분위수 기반 Good/Bad 분류] ---
        with torch.no_grad():
            centroids = self.clustering_model.clustering_manager.centroids  # [K, D]
            batch_centroids = centroids[stored_pseudo_labels]               # [B, D]
            centroid_distances = torch.norm(features - batch_centroids, p=2, dim=1)  # [B]

            good_mask = torch.zeros_like(centroid_distances, dtype=torch.bool)

            # 클러스터별로 거리의 75% 분위수(quantile=0.75)를 기준으로 분류
            num_clusters = self.clustering_model.clustering_manager.num_clusters
            for k in range(num_clusters):
                cluster_mask = stored_pseudo_labels == k
                if cluster_mask.any():
                    cluster_dists = centroid_distances[cluster_mask]
                    threshold = torch.quantile(cluster_dists, 0.75)
                    good_mask[cluster_mask] = cluster_dists < threshold

            bad_mask = ~good_mask
            bad_indicator = bad_mask.long()

            # --- 거리 정보 저장 ---
            distance_info = {
                'centroid_distances': centroid_distances.cpu(),
                'pseudo_labels': stored_pseudo_labels.cpu(),
            }

            # 메모리 업데이트
            self.clustering_model.clustering_manager.update_samples_memory(idx, features)

            # 손실용 pseudo label 확보
            loss_labels = stored_pseudo_labels.to(self.device)
            mask_new = (loss_labels == -1)
            if mask_new.any():
                loss_labels[mask_new] = torch.argmax(scores_original[mask_new], dim=1)
        # --- [수정된 로직 끝] ---


        # 클래스 가중치 계산
        class_weights = self.clustering_model.clustering_manager.compute_class_weights()


        # --- [수정된 Loss 로직 시작] ---
        
        # [수정 1] 'Live' 텐서로 초기화 (RuntimeError 방지)
        # scores_original은 clustering_model의 출력이므로 항상 grad_fn을 가짐
        loss_cluster = (scores_original.sum() * 0.0) 
        
        if self.epoch < self.hparams.threshold_epoch:
            # --- [1A. 웜업(Warm-up) 단계 Loss] ---
            # 모든 샘플에 대해 ODC Loss를 계산 (Bad 샘플 학습 방치 방지)
            loss_cluster = F.cross_entropy(
                scores_original,  # [수정] 원본 로짓 사용, F.normalize 삭제
                loss_labels,      
                weight=class_weights.to(self.device)
            )
        else:
            # --- [1B. 교정(Refinement) 단계 Loss] ---
            # 'Good' 샘플에 대해서만 ODC Loss를 계산 (신호 충돌 방지)
            if good_mask.any(): # [수정] NaN 방지 (find_unused_parameters=True 가정)
                loss_cluster = F.cross_entropy(
                    scores_original[good_mask],  # [수정] 원본 로짓 + Good 샘플
                    loss_labels[good_mask],
                    weight=class_weights.to(self.device)
                )
        # --- [수정된 Loss 로직 끝] ---

        # ODC 손실 계산 (cross-entropy) - 저장된 pseudo label과 클래스 가중치 사용    
        # loss_cluster = F.cross_entropy(scores, loss_labels, weight=class_weights.to(self.device))

        pseudo_labels_all = self.all_gather(loss_labels).view(-1)

        output = {
                'features': features.detach(),
                'labels': labels.detach(),
                'predicted_labels': loss_labels.detach(),
                'distance_info': distance_info,
                'bad': bad_indicator.detach()
            }
        self.training_steps_outputs.append(output)
        
        # --- 2. 에포크(Epoch) 기반의 조건부 로직 ---
        # self.epoch을 사용하여 현재 에포크를 확인합니다.
        if self.epoch < self.hparams.threshold_epoch:
            return loss_cluster
        
       # --- Decompose 단계 ---
        model_output = self.video_model(videos, flows=flows)  # flows 제거
        v_appearance = model_output['v_appearance']
        v_scene = model_output['v_scene']
        v_object = model_output['v_object']

        sensor_output = self.sensor_model(sensors)
        sensor_emb = sensor_output['emb']


        # --- Cross-Modal Pseudo Label Refinement ---
        loss_video_supervised = torch.tensor(0.0, device=self.device)
        loss_sensor_guided = torch.tensor(0.0, device=self.device)

        # 1. 좋은 샘플: 센서 → 비디오 (Classification 학습)
        if good_mask.any(): 
            video_logits_good = self.video_classifier(v_appearance[good_mask])
            loss_video_supervised = F.cross_entropy(
                video_logits_good,
                loss_labels[good_mask]
            )

        # 2. 나쁜 샘플: 비디오 → 센서 (Pseudo-guided refinement)
        if bad_mask.any() and self.epoch >= self.hparams.threshold_epoch + self.hparams.video_classifier_epoch:
            with torch.no_grad():
                video_logits_bad = self.video_classifier(v_appearance[bad_mask])
                video_probs_bad = F.softmax(video_logits_bad, dim=1)
                video_max_probs, video_pseudo_bad = torch.max(video_probs_bad, dim=1)
                high_confidence_mask = video_max_probs >= 0.9

            if high_confidence_mask.any():
                loss_sensor_guided = F.cross_entropy(
                    scores_original[bad_mask][high_confidence_mask],
                    video_pseudo_bad[high_confidence_mask]
                )


        # --- Sensor-guided Alignment ---
        v_appearance_all = self.all_gather(v_appearance).view(-1, v_appearance.shape[-1])
        sensor_emb_all = self.all_gather(sensor_emb).view(-1, sensor_emb.shape[-1])
        pseudo_labels_all = self.all_gather(loss_labels).view(-1)

        # contrastive loss는 sensor ↔ appearance 간 단일 alignment로 변경
        loss_align = self._calculate_alignment_loss(v_appearance_all, sensor_emb_all, pseudo_labels_all)


        # --- 최종 손실 계산 ---
        lambda_cluster = 0.0
        lambda_align = 0.0
        lambda_video_sup = 1.0
        lambda_sensor_guide = 1.5

        final_loss = (
            lambda_cluster * loss_cluster +
            lambda_align * loss_align +
            lambda_video_sup * loss_video_supervised +
            lambda_sensor_guide * loss_sensor_guided
        )

         # --- [추가] Video classifier 성능 측정 (mapping 기반) ---
        with torch.no_grad():
            # 1. video classifier 출력
            video_logits_all = self.video_classifier(v_appearance)
            video_preds = torch.argmax(video_logits_all, dim=1)

            # 2. mapping 적용 (cluster_id → real label)
            mapping = getattr(self.clustering_model.clustering_manager, "mapping", None)
            if mapping is not None and len(mapping) > 0:
                mapped_preds = torch.tensor(
                    [mapping.get(int(p.item()), int(p.item())) for p in video_preds],
                    device=self.device
                )
            else:
                mapped_preds = video_preds  # mapping이 없을 경우 fallback

            # 3. 실제 라벨과 비교
            video_labels = labels.to(self.device)
            acc_video = (mapped_preds == video_labels).float().mean()

            if self.global_rank == 0:
                # 4. wandb에 로깅
                wandb.log({"train/video_classifier_acc_mapped": acc_video})
        
        stage2_start = (
            self.hparams.threshold_epoch +
            self.hparams.video_classifier_epoch +
            self.hparams.bad_correction_epoch
        )
        if self.epoch >= stage2_start: 
            v_motion = model_output["v_motion"]
            self.training_steps_outputs[-1]["v_motion"] = v_motion
            v_app = model_output["v_appearance"]
            
            sensor_motion_emb = self.sensor_motion_model(sensors)["emb"]
            self.training_steps_outputs[-1]["s_motion"]=sensor_motion_emb

           # -------------------------------------------------------------
            # (align_loss_fn이 CovarianceAlignmentLoss라고 가정)
            align_loss = self.mutual_information_loss_fn(v_motion, sensor_motion_emb)

            # =================================================================
            # 7) [NEW] 모멘텀 기반 Contrastive Loss
            # =================================================================
            
            # --- 7-A: 온라인 복합 특징 (학습 대상) ---
            # v_app과 features는 그래디언트 차단
            
            # =================================================================
            # 7) [NEW] 모멘텀 기반 Contrastive Loss
            # =================================================================
            # --- 7-A: 온라인 복합 특징 (학습 대상) ---
            v_app_norm = F.normalize(v_app.detach(), dim=1)
            features_norm = F.normalize(features.detach(), dim=1)

            v_motion_norm = F.normalize(v_motion.detach(), dim=1)
            sensor_motion_norm = F.normalize(sensor_motion_emb.detach(), dim=1)
            # v_app과 features는 그래디언트 차단
            z_video_online = v_app_norm + v_motion_norm
            z_sensor_online = features_norm + sensor_motion_norm
            # z_video_online = torch.cat([v_app.detach(), v_motion], dim=1)
            # z_sensor_online = torch.cat([features.detach(), sensor_motion_emb], dim=1)
            self.training_steps_outputs[-1]["z_video"] = z_video_online
            self.training_steps_outputs[-1]["z_sensor"] = z_sensor_online
            # 정규화 (Cosine Similarity 계산용)
            z_video_online = F.normalize(z_video_online, dim=1)
            z_sensor_online = F.normalize(z_sensor_online, dim=1)
            # --- 7-B: 모멘텀 특징 (False Negative 감지용) ---
              # --- 7-B: 모멘텀 특징 (False Negative 감지용) ---
            with torch.no_grad():
                # (batch에서 videos, sensors를 다시 사용)
                v_motion_mom = self.momentum_video_model(videos, flows)["v_motion"]
                sensor_motion_mom = self.momentum_sensor_motion_model(sensors)["emb"]
                # 정규화
                v_motion_mom = F.normalize(v_motion_mom, dim=1)
                sensor_motion_mom = F.normalize(sensor_motion_mom, dim=1)
            # --- 7-C: DDP All-Gather ---
            # 모든 GPU의 특징과 레이블을 수집 (N = B * num_gpus)
            gathered_z_video = self.all_gather(z_video_online).reshape(-1, z_video_online.shape[-1])
            gathered_z_sensor = self.all_gather(z_sensor_online).reshape(-1, z_sensor_online.shape[-1])
            gathered_v_mom = self.all_gather(v_motion_mom).reshape(-1, v_motion_mom.shape[-1])
            gathered_s_mom = self.all_gather(sensor_motion_mom).reshape(-1, sensor_motion_mom.shape[-1])
            gathered_v_app = self.all_gather(v_app_norm).reshape(-1, v_app_norm.shape[-1])
            gathered_features = self.all_gather(features_norm).reshape(-1, features_norm.shape[-1])
            N = gathered_z_video.shape[0] # Effective Batch Size
            # --- 7-D: 가중치 행렬 계산 (W = W_app * W_motion_damp) ---
            # 1. W_app (Appearance 가중치) [수정됨]
            # Intra-modal 유사도 (Video-Video, Sensor-Sensor)를 계산
            print("v app shape", v_app_norm.shape)
            print("gathered v app shape", gathered_v_app.shape )
            sim_app_vid = gathered_v_app @ gathered_v_app.T
            sim_app_sen = gathered_features @ gathered_features.T
            # 두 유사도를 평균내어 'Appearance 유사도'로 사용
            # (유사도가 음수일 수 있으므로 F.relu로 0 이상 값만 사용)
            sim_app_max = torch.max(F.relu(sim_app_vid), F.relu(sim_app_sen))
            # 연속적인 유사도(0~1)를 가중치(1.0 ~ lambda_hard)로 변환
            # sim=0 -> W=1.0, sim=1 -> W=lambda_hard
            W_app = 1.0 + (self.hparams.lambda_hard - 1.0) * sim_app_max
            # 2. W_motion_damp (모션 감쇠 가중치) [기존과 동일]
            sim_vid_mom = gathered_v_mom @ gathered_v_mom.T
            sim_sen_mom = gathered_s_mom @ gathered_s_mom.T
            sim_stable = (sim_vid_mom + sim_sen_mom) / 2.0
            motion_damp_factor = torch.tanh(
                F.relu(sim_stable) / self.hparams.motion_damp_temp
            )
            W_conditional_damp = 1.0 - (sim_app_max * motion_damp_factor)
            # 3. W_final (최종 가중치) 및 대각선 마스킹 [기존과 동일]
            W_final = W_app * W_conditional_damp
            identity = torch.eye(N, device=self.device, dtype=torch.bool)
            W_final = W_final.masked_fill(identity, 0.0)
            # --- 7-E: InfoNCE 손실 계산 (양방향) ---
            # 1. 유사도 행렬 (Logits)
            sim_v2s = gathered_z_video @ gathered_z_sensor.T
            sim_s2v = sim_v2s.T
            logits_v2s = sim_v2s / self.hparams.contrastive_temp # (e.g., 0.07)
            logits_s2v = sim_s2v / self.hparams.contrastive_temp
            # 2. Positive / Negative 분리
            pos_v2s = logits_v2s.diag()
            pos_s2v = logits_s2v.diag()
            # 대각선(Positive)을 -inf로 마스킹
            neg_logits_v2s = logits_v2s.masked_fill(identity, -torch.inf)
            neg_logits_s2v = logits_s2v.masked_fill(identity, -torch.inf)
            # 3. 가중치가 적용된 네거티브 합 계산
            # W_final은 대칭이므로 (W_final.T == W_final) 동일하게 사용
            weighted_neg_v2s = (W_final * torch.exp(neg_logits_v2s)).sum(dim=1)
            weighted_neg_s2v = (W_final * torch.exp(neg_logits_s2v)).sum(dim=1)
            # 4. 양방향 손실 계산
            denominator_v2s = torch.exp(pos_v2s) + weighted_neg_v2s
            denominator_s2v = torch.exp(pos_s2v) + weighted_neg_s2v
            loss_v2s = -torch.log(torch.exp(pos_v2s) / denominator_v2s).mean()
            loss_s2v = -torch.log(torch.exp(pos_s2v) / denominator_s2v).mean()
            custom_contrastive_loss = (loss_v2s + loss_s2v) / 2.0
            lambda_align = 1.0
            lambda_contrastive = 1.0
            if self.global_rank == 0:
                wandb.log({"train/algin loss": align_loss,
                          "train/contrastive loss": custom_contrastive_loss})
            # --- 7-F: 최종 손실 결합 ---
            final_loss = (
                lambda_align * align_loss +
                lambda_contrastive * custom_contrastive_loss
            )
            # -------------------------------------------------------------
            # 6) 종료
            # -------------------------------------------------------------
            return final_loss
            

        if (self.global_step % 200) == 0 and self.global_rank == 0:
            try:
                # log_video_recon_grid(
                #     videos, 
                #     videos-videos.mean(dim=1, keepdim=True), 
                #     logger=self.logger, 
                #     step=self.global_step,
                # )
                # log_video_recon_grid(
                #     videos, 
                #     videos-videos.mean(dim=2, keepdim=True), 
                #     logger=self.logger, 
                #     step=self.global_step,
                #     title="dim=2"
                # )
                log_video_recon_grid(
                    video_target_ds, 
                    model_output["fused_object"], 
                    logger=self.logger, 
                    step=self.global_step,
                    title="object_rec"
                )
                log_video_recon_grid(
                    video_target_ds, 
                    model_output["fused_scene"], 
                    logger=self.logger, 
                    step=self.global_step,
                    title="scene_rec"
                )
            except Exception as e:
                print(f"[viz] skip due to error: {e}")

        if (self.global_step % 500) == 0 and self.global_rank == 0:
            try:
                try:
                    log_optical_flow_overlay_to_wandb(
                        video=videos[0].unsqueeze(0),      # [1, 20, 3, 224, 224]
                        flows=flows[0].unsqueeze(0),       # [1, 20, 2, 224, 224]
                        wandb_key="Flow_Overlay/S1_ADL1",
                        stride=8,
                        scale=6,
                        fps=10
                    )
                except Exception as e:
                    print("skip: fail to logging optical flow", e)
                log_video_recon_gif(
                    videos,
                    videos-videos.mean(dim=1, keepdim=True),
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4
                )
                log_video_recon_gif(
                    videos,
                    videos-videos.mean(dim=2, keepdim=True),
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="dim=2 video"
                ),
                log_video_recon_gif(
                    videos,
                    model_output["vis_v"],
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="no light"
                )
                # log_video_recon_gif(
                #     video_target_ds,
                #     model_output["fused_motion"],
                #     logger=self.logger,
                #     step=self.global_step,
                #     max_n=2,
                #     fps=4,
                #     title="motion_rec"
                # )
                temporal_diff_recon = to_vis_motion(temporal_diff_recon)
                temporal_diff_target = to_vis_motion(temporal_diff_target)
                log_video_recon_gif(
                    temporal_diff_target,
                    temporal_diff_recon,
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="temporal_diff"
                )
                print("motion start")


                # 시각화용 (정규화 + 3채널 복제)
                def normalize_for_viz(x):
                    B, T, C, H, W = x.shape
                    x = x.mean(dim=2)  # → [B, T, H, W]
                    x = (x - x.min(dim=1, keepdim=True).values.min(dim=1, keepdim=True).values) / \
                        (x.max(dim=1, keepdim=True).values.max(dim=1, keepdim=True).values + 1e-6)
                    return x.unsqueeze(2).repeat(1, 1, 3, 1, 1)     # [B, T, 3, H, W]

                # vis_motion = normalize_for_viz(motion_residual)
                # print("vis_motion shape", vis_motion.shape)
                # log_video_recon_gif(
                #     video_target_ds,
                #     motion_residual,
                #     logger=self.logger,
                #     step=self.global_step,
                #     max_n=2,
                #     fps=4,
                #     title="motion_residual"
                # )
                # log_video_recon_gif(
                #     video_target_ds,
                #     # motion_residual,
                #     vis_motion,
                #     logger=self.logger,
                #     step=self.global_step,
                #     max_n=2,
                #     fps=4,
                #     title="motion_residual_norm"
                # )
                log_video_recon_gif(
                    model_output['residual_motion'],
                    model_output['upsized_fused_app'],
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="residual answer / appearance upsized"
                )
            except Exception as e:
                print(f"[viz] skip gif log: {e}")

        return final_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        매 학습 배치(step)가 끝난 후 호출됩니다.
        training_step의 else 블록과 동일한 조건(freeze_epoch 이후)에서만
        모멘텀 인코더를 업데이트합니다.
        """
        # training_step의 분기문과 동일하게 조건부 실행
        if self.stage2_only:
            self._update_momentum_encoders()


            # epoch 종료 시 한번만 호출됨
    def on_train_epoch_end(self):    
                # return
            # 에포크가 끝난 후 epoch 업데이트
        self.eval()  
        outputs = self.training_steps_outputs
        print(f"{self.global_rank} Epoch {self.epoch} - Collected {len(outputs)} training step outputs.")
        self.epoch += 1
        self.clustering_model.update_epoch(self.epoch)

        with torch.no_grad():
            if self.epoch % self.clustering_model.centroids_update_interval == 0:
                self.clustering_model.clustering_manager.update_centroids_memory()
            
            if self.epoch % self.clustering_model.deal_with_small_clusters_interval == 0:
                self.clustering_model.clustering_manager.deal_with_small_clusters()

        if self.epoch % 1 == 0:
            print(f"\nEpoch {self.epoch}: Running ODC evaluation on training data...")
            self.clustering_model.evaluate(outputs)

        print("evaluate called on rank", self.global_rank)
        # 매 2 에폭마다 훈련 데이터셋에 대한 클러스터링 성능 평가

        self.train()  # 모델을 다시 훈련 모드로 설정
        print("Epoch end processing completed on rank", self.global_rank)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer