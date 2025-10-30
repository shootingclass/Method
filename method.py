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
from method_utils import log_video_recon_grid, log_video_recon_gif, match_target_to_recon, to_vis_motion
from visualizes import visualize_motion_heatmap

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
        self.epoch = 0
        
        self.appearance_classifier = nn.Linear(256, self.hparams.num_classes) 
    
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.success_labels=[0 for i in range(self.hparams.num_classes)]
        self.fail_labels=[0 for i in range(self.hparams.num_classes)]
        self.video_classifier = nn.Linear(self.hparams.embedding_dim, self.hparams.num_classes)
        print(self.global_rank, "Model initialized.")
        self.sensor_motion_model = SensorModel(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        # self.mutual_information_loss_fn = CovarianceAlignmentLoss()

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

        self.video_model.motion_branch_enable = True  # flag

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
                self.clustering_model.load_state_dict(state["clustering_model"])
                self.clustering_model.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_model.clustering_manager.centroids = state["centroids"].to(self.device)

                # ⚠️ 중요: buffer까지 GPU로 강제 이동
                self.clustering_model.to(self.device)

                print(f"[✔] Loaded pretrained clustering model from {ckpt_path}")
                           # ✅ Stage2-only 모드 (캐시 로드)
            if self.hparams.threshold_epoch == 0 and self.hparams.guide_start_epoch == 0:
                ckpt_path = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/caches/vision_model_stage1_epoch=20.pt"
                print(f"[Stage2-Only Mode] Loading pretrained clustering model from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                self.clustering_model.load_state_dict(state["cluster_model"])
                self.clustering_model.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_model.clustering_manager.centroids = state["centroids"].to(self.device)
                self.clustering_model.to(self.device)

                # VisionModel 구조에 맞게 불러오기
                vision_state_dict = state.get("vision_model", state)
                missing, unexpected = self.video_model.load_state_dict(vision_state_dict, strict=False)

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
        
        if self.epoch == self.hparams.threshold_epoch + self.hparams.guide_start_epoch + self.hparams.motion_epoch:
            stage1_ckpt_path = os.path.join(self.hparams.cache_dir, f"vision_model_stage1_epoch={self.epoch}.pt")
            torch.save({
                "cluster_model": self.clustering_model.state_dict(),
                "memory": self.clustering_model.clustering_manager.feature_bank,
                "centroids": self.clustering_model.clustering_manager.centroids,
                "vision_model": self.video_model.state_dict()},
            stage1_ckpt_path)
            print(f"[✔] Saved VisionModel weights → {stage1_ckpt_path}")


    def training_step(self, batch, batch_idx):
        # -------------------------------------------------------------
        # 0) Batch 준비
        # -------------------------------------------------------------
        videos, sensors, labels, sample_ids = batch
        idx, sample_id = sample_ids
        B = videos.size(0)

        # -------------------------------------------------------------
        # 1) 클러스터링 (ODC) 단계
        # -------------------------------------------------------------
        imu_data = self.clustering_model.augment_imu_data(sensors) \
            if self.clustering_model.epoch > 1 else sensors

        scores, features, _ = self.clustering_model(
            imu_data, return_features=True, labels=labels, idx=idx
        )
        scores = scores.clone()
        stored_pseudo_labels = self.clustering_model.get_pseudo_labels(idx)

        # --- Good/Bad 분류 (클러스터별 75% 분위수) ---
        with torch.no_grad():
            centroids = self.clustering_model.clustering_manager.centroids
            batch_centroids = centroids[stored_pseudo_labels]
            centroid_distances = torch.norm(features - batch_centroids, p=2, dim=1)
            good_mask = torch.zeros_like(centroid_distances, dtype=torch.bool)
            num_clusters = self.clustering_model.clustering_manager.num_clusters

            for k in range(num_clusters):
                cmask = (stored_pseudo_labels == k)
                if cmask.any():
                    dists = centroid_distances[cmask]
                    thr = torch.quantile(dists, 0.75)
                    good_mask[cmask] = dists < thr

            bad_mask = ~good_mask
            bad_indicator = bad_mask.long()

            distance_info = {
                "centroid_distances": centroid_distances.cpu(),
                "pseudo_labels": stored_pseudo_labels.cpu(),
            }
            self.clustering_model.clustering_manager.update_samples_memory(idx, features)

            loss_labels = stored_pseudo_labels.to(self.device)
            mask_new = (loss_labels == -1)
            if mask_new.any():
                loss_labels[mask_new] = torch.argmax(scores[mask_new], dim=1)

        class_weights = self.clustering_model.clustering_manager.compute_class_weights()

        # --- ODC loss (warm-up: 전체 / refinement: good만) ---
        loss_cluster = (scores.sum() * 0.0)
        if self.epoch < self.hparams.threshold_epoch:
            loss_cluster = F.cross_entropy(scores, loss_labels, weight=class_weights.to(self.device))
        else:
            if good_mask.any():
                loss_cluster = F.cross_entropy(
                    scores[good_mask], loss_labels[good_mask], weight=class_weights.to(self.device)
                )

        self.training_steps_outputs.append({
            "features": features.detach(),
            "labels": labels.detach(),
            "predicted_labels": loss_labels.detach(),
            "distance_info": distance_info,
            "bad": bad_indicator.detach(),
        })

        if self.epoch < self.hparams.threshold_epoch:
            return loss_cluster

        # -------------------------------------------------------------
        # 2) Decompose 단계 (비디오 & 센서)
        # -------------------------------------------------------------
        model_output = self.video_model(videos)
        v_app = model_output["v_appearance"]
        v_scene = model_output["v_scene"]
        v_object = model_output["v_object"]
        sensor_emb = self.sensor_model(sensors)["emb"]

        # Appearance 만으로 reconstruct
        fused = torch.cat([v_app, torch.zeros_like(model_output["v_motion"])], dim=1)

        # Video reconstruction
        video_recon = self.video_model.decoder(fused)
        video_target = videos                     # Ground truth
        
        # print("video recon shape", video_recon.shape)
        # print("video target shape", video_target.shape)
        # --- Before L1 loss ---
        # [B, 3, T, H, W] → [B, T, C, H, W]로 변환
        video_recon = video_recon.permute(0, 2, 1, 3, 4)

        # 해상도를 원본과 동일하게 보간

        # ✅ 원본을 recon 사이즈로 다운샘플해서 비교
        video_target_ds = match_target_to_recon(video_target, video_recon)

        loss_recon = F.l1_loss(video_recon, video_target_ds)


        # -------------------------------------------------------------
        # 3) Cross-modal pseudo label refinement
        # -------------------------------------------------------------
        loss_video_sup = torch.tensor(0.0, device=self.device)
        loss_sensor_guide = torch.tensor(0.0, device=self.device)

        if good_mask.any():
            video_logits_good = self.video_classifier(v_app[good_mask])
            loss_video_sup = F.cross_entropy(video_logits_good, loss_labels[good_mask])

        if bad_mask.any() and self.epoch >= self.hparams.threshold_epoch + self.hparams.guide_start_epoch:
            with torch.no_grad():
                video_logits_bad = self.video_classifier(v_app[bad_mask])
                prob_bad = F.softmax(video_logits_bad, dim=1)
                conf_bad, pseudo_bad = prob_bad.max(dim=1)
                hi_conf = conf_bad >= 0.9
            if hi_conf.any():
                loss_sensor_guide = F.cross_entropy(scores[bad_mask][hi_conf], pseudo_bad[hi_conf])

        # -------------------------------------------------------------
        # 4) Sensor ↔ Appearance alignment (contrastive 등)
        # -------------------------------------------------------------
        v_app_all = self.all_gather(v_app).view(-1, v_app.shape[-1])
        sensor_all = self.all_gather(sensor_emb).view(-1, sensor_emb.shape[-1])
        pseudo_all = self.all_gather(loss_labels).view(-1)

        loss_align = self._calculate_alignment_loss(v_app_all, sensor_all, pseudo_all)

        # -------------------------------------------------------------
        # 5) 1차 손실 결합
        # -------------------------------------------------------------
        lambda_cluster = 0.0
        lambda_align = 0.0
        lambda_video_sup = 1.0
        lambda_sensor_guide = 1.5
        lambda_appear_rec = 1.0

        final_loss = (
            lambda_cluster * loss_cluster +
            lambda_align * loss_align +
            lambda_video_sup * loss_video_sup +
            lambda_sensor_guide * loss_sensor_guide +
            lambda_appear_rec * loss_recon
        )

        with torch.no_grad():
            video_logits_all = self.video_classifier(v_app)
            video_preds = torch.argmax(video_logits_all, dim=1)
            self.training_steps_outputs[-1]["video_preds"] = video_preds.detach()

        # -------------------------------------------------------------
        # 6) Stage2: Video Reconstruction + Orthogonal Loss
        # -------------------------------------------------------------
        stage2_start = (
            self.hparams.threshold_epoch +
            self.hparams.guide_start_epoch +
            self.hparams.motion_epoch
        )

        if self.epoch >= stage2_start:
            if self.epoch == stage2_start:
                self.enable_stage2()

            out = model_output
            self.training_steps_outputs[-1]["v_motion"] = out["v_motion"]
            
            # print("video recon shape", video_recon.shape)
            # print("video target shape", video_target.shape)
            # --- Before L1 loss ---
            # [B, 3, T, H, W] → [B, T, C, H, W]로 변환
        

            # --- Orthogonal loss ---
            v_app_n = F.normalize(out['v_appearance'], p=2, dim=1)
            v_mot_n = F.normalize(out['v_motion'], p=2, dim=1)
            loss_ortho = (v_app_n * v_mot_n).sum(dim=1).abs().mean()

            # --- Motion-guided saliency consistency (새로운 term) ---

            # --- Temporal consistency ---
            temporal_diff_recon  = video_recon[:, :, 1:] - video_recon[:, :, :-1]
            temporal_diff_target = video_target_ds[:, :, 1:] - video_target_ds[:, :, :-1]
            loss_temporal = F.l1_loss(temporal_diff_recon, temporal_diff_target)
            # --- Combine ---
            loss_video_recon = 1.0 * loss_recon + 0.5 * loss_temporal


            # --- Saliency alignment (선택적)
            motion_sal = model_output["motion_target"]
            motion_sal = (motion_sal - motion_sal.min()) / (motion_sal.max() - motion_sal.min() + 1e-6)
            with torch.no_grad():
                app_feat_map = self.video_model.shared_encoder(videos[:, 0])  # 첫 프레임 기반
            saliency_loss = (app_feat_map.mean() * (1 - motion_sal.mean())).mean()


            saliency_loss = torch.tensor(0.0, device=self.device)
            motion_sal = out['motion_target']

            if motion_sal is not None:
                motion_sal = (motion_sal - motion_sal.min()) / (motion_sal.max() - motion_sal.min() + 1e-6)
                with torch.no_grad():
                    app_feat_map = out['v_appearance'].unsqueeze(-1).unsqueeze(-1)
                    print("app_feat_map shape", app_feat_map.shape)
                    print("motion sal shape", motion_sal.shape)

                saliency_loss = (app_feat_map.mean() * (1 - motion_sal.mean())).mean()
                final_loss = final_loss + 1.0 * loss_video_recon + 5.0 * loss_ortho + 0.5 * saliency_loss
            else:
                print("motion target is None")

            # motion residual target = original - appearance reconstruction

            fused_app = model_output["fused_app"]
            fused_app = fused_app.permute(0,2,1,3,4)
            motion_residual = (video_target_ds - fused_app)

            motion_target = motion_residual.detach()

            # motion branch reconstruction
            motion_rec = model_output["fused_motion"]
            motion_rec = motion_rec.permute(0,2,1,3,4)

            # L1 loss between motion_rec and motion_target
            loss_motion_residual = F.l1_loss(motion_rec, motion_target)

            # combine
            lambda_motion_residual = 5.0
            final_loss = final_loss + lambda_motion_residual * loss_motion_residual


            self.log_dict({
                'train/loss_video_recon': loss_video_recon,
                'train/loss_ortho': loss_ortho,
                'train/loss_saliency': saliency_loss,
            }, sync_dist=True)
            
        if self.epoch >= stage2_start + 50: 
            v_motion = model_output["v_motion"]
            v_app = model_output["v_app"]
            sensor_motion_emb = self.sensor_motion_model(sensors)
           # -------------------------------------------------------------
            # (align_loss_fn이 CovarianceAlignmentLoss라고 가정)
            align_loss = self.mutual_information_loss_fn(v_motion, sensor_motion_emb)

            # =================================================================
            # 7) [NEW] 모멘텀 기반 Contrastive Loss
            # =================================================================
            
            # --- 7-A: 온라인 복합 특징 (학습 대상) ---
            # v_app과 features는 그래디언트 차단
            z_video_online = torch.cat([v_app.detach(), v_motion], dim=1)
            z_sensor_online = torch.cat([features.detach(), sensor_motion_emb], dim=1)
            
            # 정규화 (Cosine Similarity 계산용)
            z_video_online = F.normalize(z_video_online, dim=1)
            z_sensor_online = F.normalize(z_sensor_online, dim=1)

            # =================================================================
            # 7) [NEW] 모멘텀 기반 Contrastive Loss
            # =================================================================
            # --- 7-A: 온라인 복합 특징 (학습 대상) ---
            v_app_norm = F.normalize(v_app.detach(), dim=1)
            features_norm = F.normalize(features.detach(), dim=1)
            # v_app과 features는 그래디언트 차단
            z_video_online = torch.cat([v_app.detach(), v_motion], dim=1)
            z_sensor_online = torch.cat([features.detach(), sensor_motion_emb], dim=1)
            # 정규화 (Cosine Similarity 계산용)
            z_video_online = F.normalize(z_video_online, dim=1)
            z_sensor_online = F.normalize(z_sensor_online, dim=1)
            # --- 7-B: 모멘텀 특징 (False Negative 감지용) ---
            with torch.no_grad():
                # (batch에서 videos, sensors를 다시 사용)
                v_motion_mom = self.momentum_video_model(videos)["v_motion"]
                sensor_motion_mom = self.momentum_sensor_motion_model(sensors)["emb"]
                # 정규화
                v_motion_mom = F.normalize(v_motion_mom, dim=1)
                sensor_motion_mom = F.normalize(sensor_motion_mom, dim=1)
            # --- 7-C: DDP All-Gather ---
            # 모든 GPU의 특징과 레이블을 수집 (N = B * num_gpus)
            gathered_z_video = self.all_gather(z_video_online)
            gathered_z_sensor = self.all_gather(z_sensor_online)
            gathered_v_mom = self.all_gather(v_motion_mom)
            gathered_s_mom = self.all_gather(sensor_motion_mom)
            gathered_v_app = self.all_gather(v_app_norm)
            gathered_features = self.all_gather(features_norm)
            N = gathered_z_video.shape[0] # Effective Batch Size
            # --- 7-D: 가중치 행렬 계산 (W = W_app * W_motion_damp) ---
            # 1. W_app (Appearance 가중치) [수정됨]
            # Intra-modal 유사도 (Video-Video, Sensor-Sensor)를 계산
            sim_app_vid = gathered_v_app @ gathered_v_app.T
            sim_app_sen = gathered_features @ gathered_features.T
            # 두 유사도를 평균내어 'Appearance 유사도'로 사용
            # (유사도가 음수일 수 있으므로 F.relu로 0 이상 값만 사용)
            sim_app_avg = F.relu((sim_app_vid + sim_app_sen) / 2.0)
            # 연속적인 유사도(0~1)를 가중치(1.0 ~ lambda_hard)로 변환
            # sim=0 -> W=1.0, sim=1 -> W=lambda_hard
            W_app = 1.0 + (self.hparams.lambda_hard - 1.0) * sim_app_avg
            # 2. W_motion_damp (모션 감쇠 가중치) [기존과 동일]
            sim_vid_mom = gathered_v_mom @ gathered_v_mom.T
            sim_sen_mom = gathered_s_mom @ gathered_s_mom.T
            sim_stable = (sim_vid_mom + sim_sen_mom) / 2.0
            W_motion_damp = 1.0 - torch.tanh(
                F.relu(sim_stable) / self.hparams.motion_damp_temp
            )
            # 3. W_final (최종 가중치) 및 대각선 마스킹 [기존과 동일]
            W_final = W_app * W_motion_damp
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
            # --- 7-F: 최종 손실 결합 ---
            final_loss = (
                lambda_align * align_loss +
                lambda_contrastive * custom_contrastive_loss
            )


        # ----------------------------------


        if (self.global_step % 200) == 0 and self.global_rank == 0:
            print("log video recon!!!!")
            try:
                log_video_recon_grid(
                    video_target_ds, 
                    model_output["video_recon"], 
                    logger=self.logger, 
                    step=self.global_step,
                )
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

        if (self.global_step % 200) == 0 and self.global_rank == 0:
            try:
                log_video_recon_gif(
                    video_target_ds,
                    model_output["video_recon"],
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4
                )
                log_video_recon_gif(
                    video_target_ds,
                    model_output["fused_app"],
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="app_rec"
                )
                log_video_recon_gif(
                    video_target_ds,
                    model_output["fused_motion"],
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="motion_rec"
                )
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

                vis_motion = normalize_for_viz(motion_residual)
                # print("vis_motion shape", vis_motion.shape)
                log_video_recon_gif(
                    video_target_ds,
                    motion_residual,
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="motion_residual"
                )
                log_video_recon_gif(
                    video_target_ds,
                    # motion_residual,
                    vis_motion,
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="motion_residual_norm"
                )
                log_video_recon_gif(
                    model_output['residual_motion'],
                    model_output['upsized_fused_app'],
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4,
                    title="residual"
                )
            except Exception as e:
                print(f"[viz] skip gif log: {e}")
            # 예시: training_step 안에서 한 번씩 저장
            try:
                visualize_motion_heatmap(
                    videos, 
                    model_output["motion_target"], 
                    labels=labels, 
                    idx=0,
                    save_path=f"heatmap_step{self.global_step}.png"
                )
            except Exception as e:
                print(f"[viz] skip motion heatmap: {e}")


        # -------------------------------------------------------------
        # 7) 종료
        # -------------------------------------------------------------
        return final_loss




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