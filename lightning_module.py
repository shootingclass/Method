import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import wandb

# --- 사용자 정의 모듈 임포트 ---
from model import SensorModel, VisionModel, ClusteringModel
from visualization import visualize_video


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


    # 새 훈련 에폭이 시작될 때 단 한 번 호출 --> 에포크 시작 시 clustering_model 상태 업데이트
    def on_train_epoch_start(self):
        self.clustering_model.update_epoch(self.current_epoch)


    # train_dataloader에 있는 각 미니배치(mini-batch)마다 반복적으로 호출
    def training_step(self, batch, batch_idx):

        # 데이터 준비 (Lightning이 자동으로 device로 옮겨줍니다)
        videos, sensors, _, sample_ids = batch

        # 실제 배치 크기를 텐서에서 직접 가져옵니다.
        current_batch_size = videos.size(0)
        
        # 데이터 증강 적용 (선택적)
        if self.clustering_model.epoch > 1: # 첫 에폭은 원본 데이터로 클러스터링
            imu_data_aug = self.clustering_model.augment_imu_data(sensors)
        else:
            imu_data_aug = sensors
        
        # 모델 순전파
        # 여기서 scores는 각 클러스터에 속할 확률
        scores, features = self.clustering_model(imu_data_aug, return_features=True)
        
        # sample_ids의 각 ID에 대해 과거의 메모리 뱅크를 조회
        # ID가 메모리 뱅크에 있는 경우: 해당 샘플이 이전에 어떤 클러스터로 할당되었는지 기록된 과거의 pseudo-label을 가져옴
        # ID가 메모리 뱅크에 없는 경우: 이 샘플은 학습 과정에서 처음 만나는 새로운 샘플입니다. 이 경우, "아직 할당된 적 없음"을 의미하는 특별한 값 `-1`을 반환
        stored_pseudo_labels = self.clustering_model.get_pseudo_labels(sample_ids)
        
        # 현재 모델의 예측을 바탕으로 이번 배치(batch)의 학습에 사용할 '임시 정답(pseudo-label)'을 결정하고 관리하는 역할
        # torch.no_grad의 사용 목적?
        # argmax와 같이 미분이 불가능한 연산을 수행
        # 모델의 예측 결과를 바탕으로 정답(pseudo-label)을 생성하고, 이 정답을 이용해 다시 모델을 학습시킵니다. 
        # 이때 정답을 만드는 과정 자체는 학습 대상이 되어서는 안됩니다. 
        # 만약 이 과정에 그래디언트가 흐르면, 모델은 항상 자기 예측과 똑같은 정답을 만들어 손실을 0으로 만드는 쉬운 길로 빠져버려 학습이 제대로 이루어지지 않습니다
        with torch.no_grad():
            
            # 모델이 출력한 scores를 보고, 가장 확률이 높은 클러스터의 인덱스를 현재 배치의 예측 결과로 선택
            current_cluster_ids = torch.argmax(scores, dim=1)
            
            # 신규 샘플 처리
            # 학습 과정에서 만난 모든 샘플의 라벨을 메모리 뱅크(Memory Bank)에 저장
            # stored_pseudo_labels == -1은 메모리 뱅크에 아직 저장된 적 없는 새로운 샘플을 찾아내는 조건
            # 이 새로운 샘플들에 대해서는 argmax로 얻은 현재 예측값(current_cluster_ids)을 이 샘플의 첫 pseudo-label로 할당하고 메모리 뱅크에 저장
            mask = (stored_pseudo_labels == -1)
            if mask.any():
                self.clustering_model.update_memory_bank([sid for i, sid in enumerate(sample_ids) if mask[i]], 
                                    current_cluster_ids[mask],
                                    features[mask] if features is not None else None)
                stored_pseudo_labels[mask] = current_cluster_ids[mask]

            # 기존 샘플의 점진적 업데이트            
            # 메모리 뱅크에 있는 샘플도 일정 확률로 업데이트 (ODC 논문 방식)
            # 메모리 뱅크에 이미 라벨이 있는 기존 샘플의 경우, 항상 과거의 라벨을 그대로 쓰면 클러스터 할당이 고착화될 수 있음
            # 이를 방지하기 위해, 30%의 확률로 기존 샘플의 라벨을 현재 모델의 예측값(current_cluster_ids)으로 갱신
            update_prob = 0.3  # 30% 확률로 업데이트
            device = stored_pseudo_labels.device  
            update_mask = (torch.rand(len(stored_pseudo_labels), device=device) < update_prob)
            update_mask = update_mask & ~mask  # 신규 샘플은 제외
            
            if update_mask.any():
                self.clustering_model.update_memory_bank([sid for i, sid in enumerate(sample_ids) if update_mask[i]], 
                                    current_cluster_ids[update_mask],
                                    features[update_mask] if features is not None else None)
                stored_pseudo_labels[update_mask] = current_cluster_ids[update_mask]
        
        # 클래스 가중치 계산 --> 불균형한 클러스터 문제를 해결하기 위한 장치
        # 샘플이 많은 클러스터 (다수 클러스터): 낮은 가중치를 부여
        # 샘플이 적은 클러스터 (소수 클러스터): 높은 가중치를 부여
        class_weights = self.clustering_model.clustering_manager.compute_class_weights()
        
        # ODC 손실 계산 (cross-entropy) - 저장된 pseudo label과 클래스 가중치 사용
        loss_cluster = F.cross_entropy(scores, stored_pseudo_labels, weight=class_weights.to(device))
        
        # 배치 처리 후 중심점 업데이트 - 현재 특징과 저장된 pseudo label 사용
        with torch.no_grad():
            self.clustering_model.update_centroids(features, stored_pseudo_labels)

        pseudo_labels_all = self.all_gather(stored_pseudo_labels).view(-1)
        
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
        lambda_cluster = 1.0
        lambda_ortho = 1.0
        lambda_info_nce = 1.0
        lambda_triplet = 1.0
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


    # train_dataloader의 모든 배치를 사용한 훈련이 끝나고, 해당 에폭의 훈련이 종료되었을 때 단 한 번 호출
    def on_train_epoch_end(self):

        # 매 2 에폭마다 훈련 데이터셋에 대한 클러스터링 성능 평가
        if self.current_epoch % 2 == 0 and self.current_epoch > 0:
            print(f"\nEpoch {self.current_epoch}: Running ODC evaluation on training data...")

            # 모델을 평가 모드로 설정
            self.eval()

            # evaluate_odc 호출하여 성능 지표 계산
            self.clustering_model.evaluate_odc(self.train_dataloader, self.device)

            # 모델을 다시 훈련 모드로 설정
            self.train()


    # trainer.fit()`이 호출된 직후, 실제 훈련 루프가 시작되기 바로 전에 단 한 번 호출됨
    def configure_optimizers(self):
        parameters = itertools.chain(self.video_model.parameters(), self.clustering_model.parameters())
        optimizer = optim.AdamW(parameters, lr=self.hparams.lr)
        return optimizer