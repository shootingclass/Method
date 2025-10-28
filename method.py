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
from method_utils import log_video_recon_grid, log_video_recon_gif

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

        self.training_steps_outputs = []  # 에포크 동안의 출력 저장용

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
        lambda_cluster = 1.0
        lambda_align = 0.0
        lambda_video_sup = 1.0
        lambda_sensor_guide = 1.5

        final_loss = (
            lambda_cluster * loss_cluster +
            lambda_align * loss_align +
            lambda_video_sup * loss_video_sup +
            lambda_sensor_guide * loss_sensor_guide
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
            video_recon = out['video_recon']          # [B, 3, T, H, W]
            video_target = videos                     # Ground truth
            
            # print("video recon shape", video_recon.shape)
            # print("video target shape", video_target.shape)
            # --- Before L1 loss ---
            # [B, 3, T, H, W] → [B, T, C, H, W]로 변환
            video_recon = video_recon.permute(0, 2, 1, 3, 4)

            # 해상도를 원본과 동일하게 보간
            video_recon = F.interpolate(
                video_recon.reshape(-1, 3, 56, 56),  # [B*T, 3, 56, 56]
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).reshape(B, 16, 3, 224, 224)
            # --- Reconstruction loss ---
            loss_video_recon = F.l1_loss(video_recon, video_target)

            # --- Orthogonal loss ---
            v_app_n = F.normalize(out['v_appearance'], p=2, dim=1)
            v_mot_n = F.normalize(out['v_motion'], p=2, dim=1)
            loss_ortho = (v_app_n * v_mot_n).sum(dim=1).abs().mean()

            # --- Total ---
            final_loss = final_loss + 1.0 * loss_video_recon + 5.0 * loss_ortho

            # --- Logging ---
            self.log_dict({
                'train/loss_video_recon': loss_video_recon,
                'train/loss_ortho': loss_ortho,
            }, sync_dist=True)

        if (self.global_step % 200) == 0 and self.global_rank == 0:
            try:
                log_video_recon_grid(
                    videos, 
                    model_output["video_recon"], 
                    logger=self.logger, 
                    step=self.global_step
                )
            except Exception as e:
                print(f"[viz] skip due to error: {e}")

        if (self.global_step % 500) == 0 and self.global_rank == 0:
            try:
                log_video_recon_gif(
                    videos,
                    model_output["video_recon"],
                    logger=self.logger,
                    step=self.global_step,
                    max_n=2,
                    fps=4
                )
            except Exception as e:
                print(f"[viz] skip gif log: {e}")


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