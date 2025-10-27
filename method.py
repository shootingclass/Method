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
  
        # 0. 데이터 준비 (Lightning이 자동으로 device로 옮겨줍니다)
        videos, sensors, labels, sample_ids = batch
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
        model_output = self.video_model(videos)  # flows 제거
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
        if bad_mask.any() and self.epoch >= self.hparams.threshold_epoch + self.hparams.guide_start_epoch:
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