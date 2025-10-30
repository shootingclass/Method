import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import copy

# --- 사용자 정의 모듈 임포트 ---
from model import SensorAppearanceModel, SensorMotionModel, VisionModel, ClusteringModel
from method_utils import CovarianceAlignmentLoss


####################################################################


class MethodLightningModule(pl.LightningModule):

    def __init__(self, args, datamodule=None):
        super().__init__()
        self.save_hyperparameters(args)

        self.video_model = VisionModel(latent_dim=self.hparams.embedding_dim)
        
        # =================================================================
        # [1단계-A] 모멘텀 비디오 모델 추가
        # =================================================================
        self.momentum_video_model = copy.deepcopy(self.video_model)
        for param in self.momentum_video_model.parameters():
            param.requires_grad = False
        # =================================================================

        self.sensor_appearance_model = SensorAppearanceModel(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.sensor_motion_model = SensorMotionModel(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)

        # =================================================================
        # [1단계-B] 모멘텀 센서 모션 모델 추가
        # =================================================================
        self.momentum_sensor_motion_model = copy.deepcopy(self.sensor_motion_model)
        for param in self.momentum_sensor_motion_model.parameters():
            param.requires_grad = False
        # =================================================================

        self.clustering_model = ClusteringModel(
            encoder=self.sensor_appearance_model,
            embedding_dim=self.hparams.embedding_dim,
            num_sensors=self.hparams.num_sensors,
            num_clusters=self.hparams.num_classes,
            datamodule=datamodule,
            top_k=self.hparams.top_k,
            prototype_cache_dir=os.path.join(self.hparams.cache_dir, "prototypes"),
            dataset_name=self.hparams.dataset_name,
            min_cluster_size=self.hparams.min_cluster_size
        )
        
        # (기존 코드)
        self.mutual_information_loss_fn = CovarianceAlignmentLoss()
        self.epoch = 0      
        self.success_labels=[0 for i in range(self.hparams.num_classes)]
        self.fail_labels=[0 for i in range(self.hparams.num_classes)]
        self.video_classifier = nn.Linear(self.hparams.embedding_dim, self.hparams.num_classes)
        print(self.global_rank, "Model initialized.")


    def freeze_model_parameters(self, model):
        """모델의 모든 파라미터를 고정(requires_grad=False)합니다."""
        if model is None:
            print("Warning: Tried to freeze a model that is None.")
            return
        for param in model.parameters():
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


    # 에포크 시작 시 clustering_model 상태 업데이트
    def on_train_epoch_start(self):
        # [기존 로직]
        if self.epoch == 0:
            print("Initializing memory bank at epoch 0...")
            self.clustering_model.init_prototypes_with_data(self.device, self.hparams.num_classes)

        # -----------------------------------------------------------------
        # [신규 로직] Stage 2: '외형(Appearance)' 관련 모델 고정
        # -----------------------------------------------------------------
        if self.epoch == self.hparams.freeze_epoch:
            print(f"\n--- Epoch {self.epoch}: ENTERING STAGE 2 (Freezing Appearance Models) ---")
            
            # 1. 센서 인코더 (클러스터링 모델) 고정
            print("Freezing: self.clustering_model")
            self.freeze_model_parameters(self.clustering_model)
            
            # 2. 비디오 외형 분류기 고정
            print("Freezing: self.video_classifier")
            self.freeze_model_parameters(self.video_classifier)
            
            # 3. 비디오 모델의 'v_app' 생성 파이프라인 전체 고정
            print("Freezing: self.video_model.shared_encoder")
            self.freeze_model_parameters(self.video_model.shared_encoder)
            
            print("Freezing: self.video_model.scene_branch")
            self.freeze_model_parameters(self.video_model.scene_branch)
            
            print("Freezing: self.video_model.object_branch")
            self.freeze_model_parameters(self.video_model.object_branch)
            
            print("Freezing: self.video_model.fuse")
            self.freeze_model_parameters(self.video_model.fuse)

            print("--- Models frozen. 'motion_branch' and 'sensor_motion_model' remain trainable. ---\n")

        # [기존 로직]
        self.training_steps_outputs = []


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


        # -------------------------------------------------------------
        # 2) Warm up 단계
        # -------------------------------------------------------------
        if self.epoch < self.hparams.threshold_epoch:
            return loss_cluster


        # -------------------------------------------------------------
        # 3) Decompose 단계 (비디오 & 센서)
        # -------------------------------------------------------------
        model_output = self.video_model(videos)
        v_app = model_output["v_appearance"]
        v_motion = model_output["v_motion"]
        sensor_motion_emb = self.sensor_motion_model(sensors)["emb"]


        # -------------------------------------------------------------
        # 4) Cross-modal pseudo label refinement
        # -------------------------------------------------------------
        if self.epoch < self.hparams.freeze_epoch:
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
            # 5) 1차 손실 결합
            # -------------------------------------------------------------
            lambda_cluster = 1.0
            lambda_video_sup = 1.0
            lambda_sensor_guide = 1.5

            final_loss = (
                lambda_cluster * loss_cluster +
                lambda_video_sup * loss_video_sup +
                lambda_sensor_guide * loss_sensor_guide
            )

            with torch.no_grad():
                video_logits_all = self.video_classifier(v_app)
                video_preds = torch.argmax(video_logits_all, dim=1)
                self.training_steps_outputs[-1]["video_preds"] = video_preds.detach()

        else:
            # -------------------------------------------------------------
            # 6) 상호 정보량 (Mutual Information) 손실
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
            gathered_labels = self.all_gather(stored_pseudo_labels) # Appearance 레이블

            N = gathered_z_video.shape[0] # Effective Batch Size

            # --- 7-D: 가중치 행렬 계산 (W = W_app * W_motion_damp) ---
        
            # 1. W_app (Appearance 가중치)
            # Appearance 레이블이 같으면 Hard Negative (가중치 증가)
            labels_row = gathered_labels.unsqueeze(1)
            labels_col = gathered_labels.unsqueeze(0)
            app_match_matrix = (labels_row == labels_col) # [N, N]
            
            W_app = torch.where(
                app_match_matrix, 
                self.hparams.lambda_hard, # (e.g., 2.0)
                1.0
            )

            # 2. W_motion_damp (모션 감쇠 가중치)
            # 모멘텀 특징의 유사도가 높으면 False Negative (가중치 0으로 감쇠)
            sim_vid_mom = gathered_v_mom @ gathered_v_mom.T
            sim_sen_mom = gathered_s_mom @ gathered_s_mom.T
            sim_stable = (sim_vid_mom + sim_sen_mom) / 2.0
            
            # 안정적인 모션 유사도가 높을수록(FN 의심), 가중치를 0에 가깝게 만듦
            W_motion_damp = 1.0 - torch.tanh(
                F.relu(sim_stable) / self.hparams.motion_damp_temp # (e.g., 0.1)
            )
            
            # 3. W_final (최종 가중치) 및 대각선 마스킹
            W_final = W_app * W_motion_damp
            identity = torch.eye(N, device=self.device, dtype=torch.bool)
            W_final = W_final.masked_fill(identity, 0.0) # 대각선(Positive)은 0

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


        # -------------------------------------------------------------
        # 6) 종료
        # -------------------------------------------------------------
        return final_loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        매 학습 배치(step)가 끝난 후 호출됩니다.
        training_step의 else 블록과 동일한 조건(freeze_epoch 이후)에서만
        모멘텀 인코더를 업데이트합니다.
        """
        # training_step의 분기문과 동일하게 조건부 실행
        if self.epoch >= self.hparams.freeze_epoch:
            self._update_momentum_encoders()


    # epoch 종료 시 한번만 호출됨
    def on_train_epoch_end(self):    
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

        self.train()  # 모델을 다시 훈련 모드로 설정
        print("Epoch end processing completed on rank", self.global_rank)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer