import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

# --- 사용자 정의 모듈 임포트 ---
from model import SensorModel, VisionModel, ClusteringModel


####################################################################


class MethodLightningModule(pl.LightningModule):
    def __init__(self, args, train_dataloader):
        super().__init__()
        
        self.args = args
        self.save_hyperparameters(args)

        self.video_model = VisionModel(image_size=224)
        self.sensor_model = SensorModel(sensor_channels=97, size_embeddings=self.hparams.embedding_dim)
        self.clustering_model = ClusteringModel(
            encoder=self.sensor_model,
            embedding_dim=self.hparams.embedding_dim,
            num_sensors=self.hparams.num_sensors,
            num_clusters=self.hparams.num_classes,
        )
        self.train_dataloader = None
    # 에포크 시작 시 clustering_model 상태 업데이트
    def on_train_epoch_start(self):
        self.clustering_model.update_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):

        # 0. 데이터 준비 (Lightning이 자동으로 device로 옮겨줍니다)
        videos, sensors, labels, sample_ids = batch

        # 실제 배치 크기를 텐서에서 직접 가져옵니다.
        current_batch_size = videos.size(0)

        # --- 1. 클러스터링 단계 ---
        # .module 접미사 없이 모델을 직접 호출합니다.
        # 모델 초기화
        
        # # 첫 에폭이거나 정해진 주기마다 또는 지연된 메모리 뱅크 업데이트가 있는 경우
        # if self.clustering_model.epoch == 0 or self.clustering_model.epoch % self.hparams.memory_bank_update_interval == 0 or self.clustering_model.pending_memory_update:
        #     # 지연된 업데이트의 경우, 재분배 후 백본이 한 에폭 학습한 후임을 명시
        #     if self.clustering_model.pending_memory_update:
        #         print(f"Performing delayed memory bank update after cluster redistribution at epoch {epoch+1}...")
        #         self.clustering_model.pending_memory_update = False
        #     else:
        #         print(f"Updating memory bank at epoch {self.clustering_model.epoch+1}...")
                
        #     # self.clustering_model.init_prototypes_with_data(dataloader, device)

        # 훈련 루프

        
        # 데이터 증강 적용 (선택적)
        if self.clustering_model.epoch > 1:  # 첫 에폭은 원본 데이터로 클러스터링
            imu_data_aug = self.clustering_model.augment_imu_data(sensors)
        else:
            imu_data_aug = sensors
        
        # 모델 순전파
        scores, features = self.clustering_model(imu_data_aug, return_features=True)
        
        # 메모리 뱅크에서 저장된 pseudo label 가져오기
        stored_pseudo_labels = self.clustering_model.get_pseudo_labels(sample_ids)
        
        # 클러스터 할당 (argmax) - ODC 논문과 유사하게 메모리 뱅크 활용
        with torch.no_grad():
            current_cluster_ids = torch.argmax(scores, dim=1)
            
            # 메모리 뱅크에 없는 샘플(-1로 표시)은 현재 계산한 값으로 업데이트
            mask = (stored_pseudo_labels == -1)
            if mask.any():
                self.clustering_model.update_memory_bank([sid for i, sid in enumerate(sample_ids) if mask[i]], 
                                    current_cluster_ids[mask],
                                    features[mask] if features is not None else None)
                stored_pseudo_labels[mask] = current_cluster_ids[mask]
            
            # 메모리 뱅크에 있는 샘플도 일정 확률로 업데이트 (ODC 논문 방식)
            # 점진적 클러스터 업데이트를 위한 것
            update_prob = 0.3  # 30% 확률로 업데이트
            device = stored_pseudo_labels.device  # mask와 동일한 장치 사용
            update_mask = (torch.rand(len(stored_pseudo_labels), device=device) < update_prob)
            update_mask = update_mask & ~mask  # 이미 업데이트된 샘플은 제외
            
            if update_mask.any():
                self.clustering_model.update_memory_bank([sid for i, sid in enumerate(sample_ids) if update_mask[i]], 
                                    current_cluster_ids[update_mask],
                                    features[update_mask] if features is not None else None)
                stored_pseudo_labels[update_mask] = current_cluster_ids[update_mask]
        
        # # 클래스 가중치 계산
        class_weights = self.clustering_model.clustering_manager.compute_class_weights()
        
        # # ODC 손실 계산 (cross-entropy) - 저장된 pseudo label과 클래스 가중치 사용
        scores = F.normalize(scores, dim=1)
        loss_cluster = F.cross_entropy(scores, stored_pseudo_labels, weight=class_weights.to(device))
        
        # 배치 처리 후 중심점 업데이트 - 현재 특징과 저장된 pseudo label 사용
        with torch.no_grad():
            self.clustering_model.update_centroids(features, stored_pseudo_labels)

        pseudo_labels_all = self.all_gather(stored_pseudo_labels).view(-1)
        
        # 통계 수집
        # # --- 2. 에포크(Epoch) 기반의 조건부 로직 ---
        # # self.current_epoch을 사용하여 현재 에포크를 확인합니다.
        # if self.current_epoch < self.hparams.threshold_epoch:
            
        #     # 최종 손실을 반환하면 Lightning이 알아서 backward 및 step을 수행합니다.
        #     return loss_cluster

        # --- 3. 분리(Disentanglement) 단계 ---
        model_output = self.video_model(videos)
        v_motion = model_output['v_motion']
        v_appearance = model_output['v_appearance']

        sensor_output = self.sensor_model(sensors)
        sensor_emb = sensor_output['emb']

        # 모든 GPU의 임베딩을 self.all_gather로 수집하고 합칩니다.
        v_motion_all = self.all_gather(v_motion).view(-1, 256)
        sensor_emb_all = self.all_gather(sensor_emb).view(-1, 256)
        v_appearance_all = self.all_gather(v_appearance).view(-1, v_appearance.shape[-1])

        # InfoNCE Loss 계산
        temperature = 0.07
        v_motion_norm = F.normalize(v_motion_all, p=2, dim=1)
        sensor_emb_norm = F.normalize(sensor_emb_all, p=2, dim=1)
        sim_matrix = torch.matmul(v_motion_norm, sensor_emb_norm.T) / temperature # 이제 (12x256) @ (256x12) 연산이 됨
        nce_labels = torch.arange(sim_matrix.size(0), device=self.device)
        info_nce_loss = (F.cross_entropy(sim_matrix, nce_labels) + F.cross_entropy(sim_matrix.T, nce_labels)) / 2

        # 직교성 제약 Loss 계산
        v_appearance_norm = F.normalize(v_appearance_all, p=2, dim=1)
        cosine_similarity = (v_motion_norm * v_appearance_norm).sum(dim=1)
        ortho_loss = (cosine_similarity ** 2).mean()

        # Triplet Loss with Hard Negative Mining
        dist_matrix = torch.cdist(v_motion_all, self.clustering_model.clustering_manager.centroids, p=2)
        positive_distances = dist_matrix.gather(1, pseudo_labels_all.unsqueeze(1)).squeeze()
        masked_dist_matrix = dist_matrix.clone()
        masked_dist_matrix.scatter_(1, pseudo_labels_all.unsqueeze(1), float('inf'))
        hard_negative_distances = torch.min(masked_dist_matrix, dim=1).values
        margin = 1.0
        triplet_loss = torch.relu(positive_distances - hard_negative_distances + margin).mean()

        # 최종 손실 계산
        lambda_cluster = 0.5
        lambda_ortho = 0.1
        lambda_info_nce = 0.9
        lambda_triplet = 0.1
        final_loss = (lambda_cluster * loss_cluster +
                    lambda_info_nce * info_nce_loss +
                    lambda_ortho * ortho_loss +
                    lambda_triplet * triplet_loss)

        # 개별 및 최종 손실을 로깅합니다.
        self.log('train/info_nce_loss', info_nce_loss, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/ortho_loss', ortho_loss, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/triplet_loss', triplet_loss, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/final_loss', final_loss, batch_size=current_batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return final_loss
    
    def on_train_end(self):
        print("Training end")
        if self.clustering_model.epoch % 2 == 0:
            self.clustering_model.evaluate_odc(self.train_dataloader, self.device)
    # Trainer가 학습을 시작할 때 자동으로 호출됨
    def configure_optimizers(self):
        parameters = itertools.chain(self.video_model.parameters(), self.clustering_model.parameters())
        optimizer = optim.AdamW(parameters, lr=self.hparams.lr)
        return optimizer