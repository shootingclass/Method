import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

# --- 사용자 정의 모듈 임포트 ---
from model import SensorModel, VisionModel, ClusteringModel


####################################################################


class MethodLightningModule(pl.LightningModule):
    def __init__(self, args):
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
            prototypes=None,
            alpha_fixed=self.hparams.alpha_fixed
        )

    # 에포크 시작 시 clustering_model 상태 업데이트
    def on_train_epoch_start(self):
        self.clustering_model.update_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):

        # 0. 데이터 준비 (Lightning이 자동으로 device로 옮겨줍니다)
        videos, sensors, labels, _ = batch

        # 실제 배치 크기를 텐서에서 직접 가져옵니다.
        current_batch_size = videos.size(0)

        # --- 1. 클러스터링 단계 ---
        # .module 접미사 없이 모델을 직접 호출합니다.
        rule_based_feature = self.clustering_model.get_representative_sensor_feature(sensors, labels, self.hparams.num_sensors)
        scores_cluster, final_feature_cluster, alpha = self.clustering_model(sensors, rule_based_feature)
        scores_sk = self.clustering_model.sinkhorn_knopp(scores_cluster)

        with torch.no_grad():
            pseudo_labels = torch.argmax(scores_sk, dim=1)

        # self.all_gather를 사용하여 모든 GPU의 텐서를 수집합니다.
        # 출력이 (world_size, batch_size, ...) 형태이므로 view/reshape로 합쳐줍니다.
        final_feature_cluster_all = self.all_gather(final_feature_cluster).view(-1, final_feature_cluster.shape[-1])
        pseudo_labels_all = self.all_gather(pseudo_labels).view(-1)

        # 글로벌 배치 기준으로 MSE Loss를 계산합니다.
        mse_loss = F.mse_loss(final_feature_cluster_all, self.clustering_model.prototypes[pseudo_labels_all])

        # Diversity Loss를 계산합니다.
        prototypes = self.clustering_model.prototypes
        n_proto = prototypes.shape[0]

        # p1/p2를 명시적으로 확장하여 브로드캐스팅 경고를 피합니다.
        p1 = prototypes.unsqueeze(1).expand(n_proto, n_proto, -1)
        p2 = prototypes.unsqueeze(0).expand(n_proto, n_proto, -1)
        mse_matrix = F.mse_loss(p1, p2, reduction='none').mean(dim=2)
        diversity_loss = - (mse_matrix.sum()) / (n_proto * (n_proto - 1))

        loss_cluster = mse_loss + diversity_loss
        self.log('train/cluster_loss', loss_cluster, batch_size=current_batch_size, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

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
        dist_matrix = torch.cdist(v_motion_all, self.clustering_model.prototypes, p=2)
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
    
    # Trainer가 학습을 시작할 때 자동으로 호출됨
    def configure_optimizers(self):
        parameters = itertools.chain(self.video_model.parameters(), self.clustering_model.parameters())
        optimizer = optim.AdamW(parameters, lr=self.hparams.lr)
        return optimizer