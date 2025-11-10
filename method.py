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
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from method_utils import gather, log_video_recon_grid, log_video_recon_gif, to_vis_motion, CovarianceAlignmentLoss, overlay_motion_heatmap, log_optical_flow_overlay_to_wandb, log_shared_feat_overlay_to_wandb
import copy
import torch.distributed as dist

# --- 사용자 정의 모듈 임포트 ---
from model import SensorEncoder, SensorMotionEncoder, SensorModel, VisionModel, ClusteringModule

####################################################################



class MethodLightningModule(pl.LightningModule):

    def __init__(self, args, datamodule=None):
        super().__init__()
        self.save_hyperparameters(args)
        # 1. 모델 구성 요소 초기화

        self.video_model = VisionModel(latent_dim=self.hparams.embedding_dim)
        self.sensor_appearance_encoder= SensorEncoder(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.sensor_motion_encoder = SensorEncoder(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.clustering_module = ClusteringModule(
            encoder=self.sensor_appearance_encoder,
            embedding_dim=self.hparams.embedding_dim,
            num_sensors=self.hparams.num_sensors,
            num_clusters=self.hparams.num_classes,
            datamodule=datamodule,
            top_k=self.hparams.top_k,
            prototype_cache_dir=os.path.join(self.hparams.cache_dir, "prototypes"),
            dataset_name=self.hparams.dataset_name,
            min_cluster_size=self.hparams.min_cluster_size,
            mid_label=self.hparams.mid_label
        )
        self.sensor_model = SensorModel(self.sensor_appearance_encoder, self.sensor_motion_encoder, self.clustering_module)


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
        ## video_model 고정!!!!
        # for param in self.video_model.parameters():
            # param.requires_grad = False
            

        self.momentum_video_model = copy.deepcopy(self.video_model)
        for param in self.momentum_video_model.parameters():
            param.requires_grad = False
        # =================================================================

                                                             
        # =================================================================
        # [1단계-B] 모멘텀 센서 모션 모델 추가
        # =================================================================
        self.momentum_sensor_model = copy.deepcopy(self.sensor_model)
        for param in self.momentum_sensor_model.parameters():
            param.requires_grad = False
        
        self.train_video_accs = []  # video classifier accuracy 저장용
        self.log_buffer = {}  # ✅ 로그 누적용 버퍼 초기화


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
        for param_q, param_k in zip(self.sensor_model.parameters(), 
                                self.momentum_sensor_model.parameters()):
            # param_k = param_k * m + param_q * (1. - m)
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def enable_stage2(self):
        """appearance는 고정, motion만 학습"""
        # 1) appearance 관련 모듈 동결 (scene/object/fuse/backbone 등)
        for n, p in self.video_model.named_parameters():
            if any(key in n for key in ["shared_encoder", "scene", "object", "fuse", "appearance", "video_classifier"]):
                p.requires_grad = False
        
        for n, p in self.sensor_model.named_parameters():
            if any(key in n for key in ["appearance"]):
                p.requires_grad = False

        # 2) 모션 브랜치만 학습
        for n, p in self.video_model.named_parameters():
            if "motion_encoder" in n:
                p.requires_grad = True
                print("video motion encoder trainable")
        
        for n, p in self.sensor_model.named_parameters():
            if "motion_encoder" in n:
                p.requires_grad = True
                print("sensor motion encoder trainable")


        self.stage2_only = True
        print("\n[🔍 Trainable Parameter Overview]\n")
        total_trainable, total_frozen = 0, 0

        for name, param in self.video_model.named_parameters():
            status = "✅ TRAINABLE" if param.requires_grad else "🚫 FROZEN"
            count = param.numel()
            print(f"{status:12s} | {name:60s} | {count:,} params")

            if param.requires_grad:
                total_trainable += count
            else:
                total_frozen += count

        print("\n" + "-" * 90)
        print(f"🟢 Total trainable params: {total_trainable:,}")
        print(f"🔴 Total frozen params:    {total_frozen:,}")
        print(f"📦 Total params:           {total_trainable + total_frozen:,}")
        print("-" * 90 + "\n")

        total_trainable, total_frozen = 0, 0
        for name, param in self.sensor_model.named_parameters():
            status = "✅ TRAINABLE" if param.requires_grad else "🚫 FROZEN"
            count = param.numel()
            print(f"{status:12s} | {name:60s} | {count:,} params")

            if param.requires_grad:
                total_trainable += count
            else:
                total_frozen += count

        print("\n" + "-" * 90)
        print(f"🟢 Total s-trainable params: {total_trainable:,}")
        print(f"🔴 Total s-frozen params:    {total_frozen:,}")
        print(f"📦 Total s-params:           {total_trainable + total_frozen:,}")
        print("-" * 90 + "\n")


    # 에포크 시작 시 clustering_module 상태 업데이트
    def on_train_epoch_start(self):

        # 첫 에폭에서 메모리 뱅크 초기화 (중요!)
        if self.epoch == 0:
            print("Initializing memory bank at epoch 0...")
            self.clustering_module.init_prototypes_with_data(self.device, self.hparams.num_classes)
            self.stage2_only = False
            if self.hparams.threshold_epoch == -1:
                ckpt_path = os.path.join(self.hparams.cache_dir, f"threshold_epoch=5.pt")
                state = torch.load(ckpt_path, map_location="cpu")
                self.clustering_module.load_state_dict(state["cluster_model"])
                self.clustering_module.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_module.clustering_manager.centroids = state["centroids"].to(self.device)
                self.clustering_module.clustering_manager.label_bank = state["label_bank"].to(self.device)

                # ⚠️ 중요: buffer까지 GPU로 강제 이동
                self.clustering_module.to(self.device)

                print(f"[✔] Loaded pretrained clustering model from {ckpt_path}")

            elif self.hparams.threshold_epoch == -2 and self.hparams.video_classifier_epoch == 2:
                ckpt_dir = self.hparams.cache_dir
                ckpt_candidates = [
                    os.path.join(ckpt_dir, f"vision_model_classifier_cetroid_threshold={self.hparams.centroid_threshold}_epoch=10.pt"),
                    os.path.join(ckpt_dir, f"vision_model_classifier_cetroid_threshold={self.hparams.centroid_threshold}_epoch=9.pt"),
                ]

                # ✅ 존재하는 체크포인트 선택
                ckpt_path = None
                for path in ckpt_candidates:
                    if os.path.exists(path):
                        ckpt_path = path
                        break

                if ckpt_path is None:
                    raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")

                print(f"[Classifier Cached Mode] Loading pretrained models from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                self.clustering_module.load_state_dict(state["cluster_model"])
                self.clustering_module.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_module.clustering_manager.centroids = state["centroids"].to(self.device)
                self.clustering_module.clustering_manager.label_bank = state["label_bank"].to(self.device)
                self.clustering_module.to(self.device)

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

            # ✅ Stage2-only 모드 (캐시 로드)
            elif self.hparams.threshold_epoch == 0 and self.hparams.video_classifier_epoch == 0 and self.hparams.bad_correction_epoch == 0:
                # ✅ 후보 경로 정의
                ckpt_dir = self.hparams.cache_dir
                ckpt_candidates = [
                    os.path.join(ckpt_dir, "start_stage2_epoch=19.pt"),
                    os.path.join(ckpt_dir, "start_stage2_epoch=15.pt"),
                ]

                # ✅ 존재하는 체크포인트 선택
                ckpt_path = None
                for path in ckpt_candidates:
                    if os.path.exists(path):
                        ckpt_path = path
                        break

                if ckpt_path is None:
                    raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")

                print(f"[Stage2-Only Mode] Loading pretrained models from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                self.clustering_module.load_state_dict(state["cluster_model"])
                self.clustering_module.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_module.clustering_manager.centroids = state["centroids"].to(self.device)
                self.clustering_module.clustering_manager.label_bank = state["label_bank"].to(self.device)
                self.clustering_module.to(self.device)

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
                self.enable_stage2()
               


        else:
            if self.hparams.save_stage_cache:
                if self.epoch == self.hparams.threshold_epoch - 1 and self.epoch > 3 and self.global_rank == 0:
                    ckpt_path = os.path.join(self.hparams.cache_dir, f"threshold_epoch={self.hparams.threshold_epoch}.pt")
                    torch.save({
                        "cluster_model": self.clustering_module.state_dict(),
                        "memory": self.clustering_module.clustering_manager.feature_bank,
                        "centroids": self.clustering_module.clustering_manager.centroids,
                        "label_bank": self.clustering_module.clustering_manager.label_bank,
                    }, ckpt_path)

                    print(f"[✔] Saved clustering stage-1 weights → {ckpt_path}")
                
                elif self.epoch == self.hparams.threshold_epoch + self.hparams.video_classifier_epoch and self.global_rank == 0:
                    classifier_ckpt_path = os.path.join(self.hparams.cache_dir, f"vision_model_classifier_cetroid_threshold={self.hparams.centroid_threshold}_epoch={self.epoch}.pt")
                    torch.save({
                        "cluster_model": self.clustering_module.state_dict(),
                        "memory": self.clustering_module.clustering_manager.feature_bank,
                        "centroids": self.clustering_module.clustering_manager.centroids,
                        "label_bank": self.clustering_module.clustering_manager.label_bank,
                        "vision_model": self.video_model.state_dict()},
                    classifier_ckpt_path)
                    print(f"[✔] Saved VisionModel weights → {classifier_ckpt_path}")

                
                elif self.epoch == self.hparams.threshold_epoch + self.hparams.video_classifier_epoch + self.hparams.bad_correction_epoch:
                    stage1_ckpt_path = os.path.join(self.hparams.cache_dir, f"start_stage2_epoch={self.epoch}.pt")
                    torch.save({
                        "cluster_model": self.clustering_module.state_dict(),
                        "memory": self.clustering_module.clustering_manager.feature_bank,
                        "centroids": self.clustering_module.clustering_manager.centroids,
                        "label_bank": self.clustering_module.clustering_manager.label_bank,
                        "vision_model": self.video_model.state_dict()},
                    stage1_ckpt_path)
                    print(f"[✔] Saved VisionModel weights → {stage1_ckpt_path}")
        self.training_steps_outputs = []  # 에포크 동안의 출력 저장용


    def training_step(self, batch, batch_idx):
  
        # 0. 데이터 준비 (Lightning이 자동으로 device로 옮겨줍니다)
        videos, sensors, labels, sample_ids, flows = batch
        idx, sample_id = sample_ids

        # 실제 배치 크기를 텐서에서 직접 가져옵니다.
        current_batch_size = videos.size(0)


        # --- 1. 클러스터링 단계 ---
        
        # 데이터 증강 적용 (선택적)
        if self.clustering_module.epoch > 1:  # 첫 에폭은 원본 데이터로 클러스터링
            imu_data_aug = self.clustering_module.augment_imu_data(sensors)
        else:
            imu_data_aug = sensors
        
        # 모델 순전파
        scores, features, _ = self.sensor_model.encoding_appearance(imu_data_aug, labels=labels, return_features=True, idx=idx)
        
        # 메모리 뱅크에서 저장된 pseudo label 가져오기
        stored_pseudo_labels = self.clustering_module.get_pseudo_labels(idx)

        scores_original = scores.clone()

        # --- [수정된 로직 시작: 클러스터별 75% 분위수 기반 Good/Bad 분류] ---
        with torch.no_grad():
            centroids = self.clustering_module.clustering_manager.centroids  # [K, D]
            batch_centroids = centroids[stored_pseudo_labels]               # [B, D]
            # centroid_distances = torch.norm(features - batch_centroids, p=2, dim=1)  # [B]
            centroid_distances = torch.sqrt(((F.normalize(features, dim=1) - batch_centroids) ** 2).sum(dim=1))

            good_mask = torch.zeros_like(centroid_distances, dtype=torch.bool)

            # 클러스터별로 거리의 75% 분위수(quantile=0.75)를 기준으로 분류
            num_clusters = self.clustering_module.clustering_manager.num_clusters
            for k in range(num_clusters):
                cluster_mask = stored_pseudo_labels == k
                if cluster_mask.any():
                    cluster_dists = centroid_distances[cluster_mask]
                    # threshold = torch.quantile(cluster_dists, self.hparams.centroid_threshold)
                    # 아직 cluster_stats가 없거나 비어 있으면 계산 먼저 실행
                    manager = self.clustering_module.clustering_manager
                    if not hasattr(manager, "cluster_stats") or len(manager.cluster_stats) == 0:
                        manager.compute_class_weights()  # 이 시점에서 cluster_stats 생성됨
                    threshold = manager.cluster_stats[k]["q75"]
                    good_mask[cluster_mask] = cluster_dists < threshold

            bad_mask = ~good_mask
            bad_indicator = bad_mask.long()

            # --- 거리 정보 저장 ---
            distance_info = {
                'centroid_distances': centroid_distances.cpu(),
                'pseudo_labels': stored_pseudo_labels.cpu(),
            }

            # 메모리 업데이트
            self.clustering_module.clustering_manager.update_samples_memory(idx, features)

            # 손실용 pseudo label 확보
            loss_labels = stored_pseudo_labels.to(self.device)
            mask_new = (loss_labels == -1)
            if mask_new.any():
                loss_labels[mask_new] = torch.argmax(scores_original[mask_new], dim=1)
        # --- [수정된 로직 끝] ---


        # 클래스 가중치 계산
        class_weights = self.clustering_module.clustering_manager.compute_class_weights()


        # --- [수정된 Loss 로직 시작] ---
        
        # [수정 1] 'Live' 텐서로 초기화 (RuntimeError 방지)
        # scores_original은 clustering_module의 출력이므로 항상 grad_fn을 가짐
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

        sensor_output = self.sensor_model.sensor_appearance_encoder(sensors)
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
                high_confidence_mask = video_max_probs >= self.hparams.threshold_classifier_confidence

                # ✅ confidence 기록용 (기존 로직 그대로)
                if self.global_rank == 0:
                    if not hasattr(self, "video_confidence_log"):
                        self.video_confidence_log = []
                    self.video_confidence_log.append(video_max_probs.detach().cpu())

            if high_confidence_mask.any():
                loss_sensor_guided = F.cross_entropy(
                    scores_original[bad_mask][high_confidence_mask],
                    video_pseudo_bad[high_confidence_mask]
                )

                # label_b

            # if high_confidence_mask.any():
            #     target_scores = scores_original[bad_mask][high_confidence_mask]
            #     target_labels = video_pseudo_bad[high_confidence_mask]
            #     loss_sensor_guided = F.cross_entropy(target_scores, target_labels)

            #     # ✅ good에도 살짝 주기 (soft supervision)
            #     if good_mask.any():
            #         video_logits_good = self.video_classifier(v_appearance[good_mask])
            #         video_probs_good = F.softmax(video_logits_good, dim=1)
            #         video_max_probs, video_pseudo_good = torch.max(video_probs_good, dim=1)
            #         high_conf_good = video_max_probs >= 0.95

            #         if high_conf_good.any():
            #             loss_sensor_guided_good = F.cross_entropy(
            #                 scores_original[good_mask][high_conf_good],
            #                 video_pseudo_good[high_conf_good]
            #             )
            #             loss_sensor_guided += 0.1 * loss_sensor_guided_good  # 가중치 낮게


        # --- 최종 손실 계산 ---
        lambda_cluster = 1.0
        lambda_video_sup = 1.0
        lambda_sensor_guide = 10.0

        final_loss = (
            lambda_cluster * loss_cluster +
            lambda_video_sup * loss_video_supervised +
            lambda_sensor_guide * loss_sensor_guided
        )

        if self.global_rank == 0:
            wandb.log({"train/loss_sensor_guide": loss_sensor_guided,
                    "train/loss_cluster": loss_cluster,
                "train/loss_video_supervised": loss_video_supervised})

         # --- [추가] Video classifier 성능 측정 (mapping 기반) ---
        with torch.no_grad():
            # 1. video classifier 출력
            video_logits_all = self.video_classifier(v_appearance)
            video_preds = torch.argmax(video_logits_all, dim=1)

            # 2. mapping 적용 (cluster_id → real label)
            mapping = getattr(self.clustering_module.clustering_manager, "mapping", None)
            if mapping is not None and len(mapping) > 0:
                mapped_video_preds = torch.tensor(
                    [mapping.get(int(p.item()), int(p.item())) for p in video_preds],
                    device=self.device
                )
            else:
                mapped_video_preds = video_preds  # mapping이 없을 경우 fallback

            # 3. 실제 라벨과 비교

            spatial_labels = self.clustering_module._remap_pairwise_7(labels.to(self.device))
            if spatial_labels == None:
                spatial_labels = labels
            acc_video = (mapped_video_preds == spatial_labels).float().mean()
            sensor_preds = torch.argmax(scores_original, dim=1)  # sensor model의 예측 (cluster id)

            # 2️⃣ cluster → real label mapping 적용
            mapping = getattr(self.clustering_module.clustering_manager, "mapping", None)
            if mapping is not None and len(mapping) > 0:
                mapped_sensor_preds = torch.tensor(
                    [mapping.get(int(p.item()), int(p.item())) for p in sensor_preds],
                    device=self.device
                )
            else:
                mapped_sensor_preds = sensor_preds

            # ✅ good/bad별 센서 성능
            acc_sensor_good = torch.tensor(0.0, device=self.device)
            acc_sensor_bad = torch.tensor(0.0, device=self.device)
            if good_mask.any():
                acc_sensor_good = (mapped_sensor_preds[good_mask] == spatial_labels[good_mask]).float().mean()
            if bad_mask.any():
                acc_sensor_bad = (mapped_sensor_preds[bad_mask] == spatial_labels[bad_mask]).float().mean()
            
            acc_sensor_all = (mapped_sensor_preds == spatial_labels).float().mean()
            # -------------------------
            # 4️⃣ 로그 버퍼에 저장
            # -------------------------
            self.log_buffer.setdefault("classifier_all", []).append(acc_video.detach().cpu())
            self.log_buffer.setdefault("acc_good", []).append(acc_sensor_good.detach().cpu())
            self.log_buffer.setdefault("acc_bad", []).append(acc_sensor_bad.detach().cpu())
            self.log_buffer.setdefault("good_ratio", []).append(good_mask.float().mean().detach().cpu())

            # -------------------------
            # 5️⃣ confusion matrix용 데이터 저장 (센서 기준)
            # -------------------------
            if not hasattr(self, "cm_data"):
                self.cm_data = {"preds": [], "labels": []}
            self.cm_data["preds"].append(mapped_video_preds.cpu())
            self.cm_data["labels"].append(spatial_labels.cpu())
        
        stage2_start = (
            self.hparams.threshold_epoch +
            self.hparams.video_classifier_epoch +
            self.hparams.bad_correction_epoch
        )
        if self.epoch >= stage2_start: 
            v_motion = model_output["v_motion"]
            # print("v motion shape", v_motion.shape)
            v_app = model_output["v_appearance"]

            
            v_app_norm = F.normalize(v_app.detach(), dim=1)
            features_norm = F.normalize(features.detach(), dim=1)

            sensor_motion_emb = self.sensor_model.encoding_motion(sensors)["emb"]

           # -------------------------------------------------------------
            # (align_loss_fn이 CovarianceAlignmentLoss라고 가정)
            align_loss = self.mutual_information_loss_fn(v_motion, sensor_motion_emb) # normalize 함 해봤음
            
            # =================================================================
            # 7) [NEW] 모멘텀 기반 Contrastive Loss
            # =================================================================
            # --- 7-A: 온라인 복합 특징 (학습 대상) ---
       
            z_video_online = model_output["z_video_online"]
            z_sensor_online = self.sensor_model(imu_data_aug)

            # 정규화 (Cosine Similarity 계산용)
            # z_video_online = F.normalize(z_video_online, dim=1)
            # z_sensor_online = F.normalize(z_sensor_online, dim=1)
  

            # --- 7-B: 모멘텀 특징 (False Negative 감지용) ---
            with torch.no_grad():
                # (batch에서 videos, sensors를 다시 사용)
                v_motion_mom = self.momentum_video_model(videos, flows)["v_motion"]
                sensor_motion_mom = self.momentum_sensor_model.encoding_motion(sensors)["emb"]

           
            print("gradient check (Mom/Onl):", sensor_motion_mom.requires_grad, z_sensor_online.requires_grad)

            # --- 7-C: DDP All-Gather ---
            # 모든 GPU의 특징과 레이블을 수집 (N = B * num_gpus)
            # gathered_z_video = self.all_gather(z_video_online).reshape(-1, z_video_online.shape[-1])
            # gathered_z_sensor = self.all_gather(z_sensor_online).reshape(-1, z_sensor_online.shape[-1])
            # gathered_v_mom = self.all_gather(v_motion_mom).reshape(-1, v_motion_mom.shape[-1])
            # gathered_s_mom = self.all_gather(sensor_motion_mom).reshape(-1, sensor_motion_mom.shape[-1])
            # gathered_v_app = self.all_gather(v_app_norm).reshape(-1, v_app_norm.shape[-1])
            # gathered_features = self.all_gather(features_norm).reshape(-1, features_norm.shape[-1])
            # gathered_z_video = gather(z_video_online)
            # gathered_z_sensor = gather(z_sensor_online)
            # gathered_v_mom = gather(v_motion_mom)
            # gathered_s_mom = gather(sensor_motion_mom)
            # gathered_v_app = gather(v_app_norm)
            # gathered_features = gather(features_norm)

            # --- 7-C: DDP All-Gather ---
            gathered_z_video = F.normalize(gather(z_video_online), dim=1)
            gathered_z_sensor = F.normalize(gather(z_sensor_online), dim=1)
            gathered_v_mom = F.normalize(gather(v_motion_mom), dim=1)
            gathered_s_mom = F.normalize(gather(sensor_motion_mom), dim=1)
            gathered_v_app = F.normalize(gather(v_app_norm), dim=1)
            gathered_features = F.normalize(gather(features_norm), dim=1)

            self.training_steps_outputs[-1]["z_video"] = F.normalize((z_video_online), dim=1)
            self.training_steps_outputs[-1]["z_sensor"] = F.normalize((z_sensor_online), dim=1)
            self.training_steps_outputs[-1]["v_motion"] = F.normalize((v_motion_mom), dim=1)
            self.training_steps_outputs[-1]["s_motion"]= F.normalize((sensor_motion_mom), dim=1)
            self.training_steps_outputs[-1]["v_appearance"] = F.normalize((v_app), dim=1)

            motion_labels = self.clustering_module._remap_pairwise_2(labels)
            motion_labels_np = gather(motion_labels).cpu().numpy()
            labels_gathered = gather(labels)
            labels_np = labels_gathered.cpu().numpy()


                        # ============================================================
            # 2️⃣ raw temporal similarity 계산 (뒤 절반을 temporal feature로 간주)
            # ============================================================
            if gathered_z_video.dim() == 2:
                # [N, D] case
                D_half = gathered_z_video.size(1) // 2
                v_temp = gathered_z_video[:, D_half:]       # temporal feature 영역
                s_temp = gathered_z_sensor[:, D_half:]
            elif gathered_z_video.dim() == 3:
                # [N, T, D] case → 시간 절반 사용 후 mean pooling
                T_half = gathered_z_video.size(1) // 2
                v_temp = gathered_z_video[:, T_half:, :].mean(dim=1)
                s_temp = gathered_z_sensor[:, T_half:, :].mean(dim=1)
            else:
                raise ValueError("Unexpected shape for z_video/sensor features")

            # ⚠️ 여기서는 정규화하지 않음 (raw similarity)
            sim_vid_raw = v_temp @ v_temp.T              # [N, N]
            sim_sen_raw = s_temp @ s_temp.T  
            sim_z_sen_raw = gathered_z_sensor @ gathered_z_sensor.T
            sim_z_vid_raw = gathered_z_video @ gathered_z_video.T





            # # print("requires_grad z_video:", z_video_online.requires_grad)  
            # # print("gathered z gradient", gathered_z_video.requires_grad)
            # N = gathered_z_video.shape[0] # Effective Batch Size
            # # --- 7-D: 가중치 행렬 계산 (W = W_app * W_motion_damp) ---
            # 1. Appearance 비중 조절 (온건한 re-weighting)
            sim_app_vid = gathered_v_app @ gathered_v_app.T
            sim_app_sen = gathered_features @ gathered_features.T
            # [수정됨] F.relu()를 torch.max() 밖으로 이동하여 음수 유사도를 먼저 처리
            sim_app_vid_rel = F.relu(sim_app_vid)
            sim_app_sen_rel = F.relu(sim_app_sen)
            sim_app_max = torch.max(sim_app_vid_rel, sim_app_sen_rel) # [N, N]
            # sim_app_mean = (sim_app_vid_rel + sim_app_sen_rel)/2 # 빠른 디버깅 위해 mean 사용!!
            # [버그 1 수정] sim_app_avg -> sim_app_max
            # W_app = sim_app_mean
            W_app = sim_app_max
            # W_app *= self.hparams.lambda_hard
            
            spatial_labels_gathered = gather(spatial_labels)
            spatial_labels_np = spatial_labels_gathered.cpu().numpy()
            #     # --- 7-D: 가중치 행렬 계산 (W = W_app * W_motion_damp) ---

              # --- 7-D: 가중치 행렬 계산 (W = W_app * W_motion_damp) ---
            # 1. Appearance 비중 조절 (온건한 re-weighting)
            sim_app_vid = gathered_v_app @ gathered_v_app.T
            sim_app_sen = gathered_features @ gathered_features.T
            # [수정됨] F.relu()를 torch.max() 밖으로 이동하여 음수 유사도를 먼저 처리
            sim_app_vid_rel = F.relu(sim_app_vid)
            sim_app_sen_rel = F.relu(sim_app_sen)
            sim_app_max = torch.max(sim_app_vid_rel, sim_app_sen_rel) # [N, N]
            # [버그 1 수정] sim_app_avg -> sim_app_max
            W_app = 1.0 + (3.0 - 1.0) * sim_app_max
            # 2. Motion 감쇠 (FN filtering)
            sim_vid_mom = gathered_v_mom @ gathered_v_mom.T
            sim_sen_mom = gathered_s_mom @ gathered_s_mom.T
            sim_cross_mom = gathered_v_mom @ gathered_s_mom.T

            sim_stable = (sim_vid_mom + sim_sen_mom) / 2.0

            # 🔹 Global Softmax Normalization (short, symmetric, stable)
            tau = 0.7
            sim_exp = torch.exp((sim_app_max*sim_stable - sim_app_max*sim_stable.max()) / tau)
            sim_stable = sim_exp / (sim_exp.sum() + 1e-6)
            sim_stable = (sim_stable - sim_stable.min()) / (sim_stable.max() - sim_stable.min() + 1e-6)

            motion_damp_factor = sim_stable
            # [버그 1 수정] sim_app_avg -> sim_app_max
            W_conditional_damp = 1.0 - sim_app_max * motion_damp_factor
            # =====================================================================
            # 3. [안정성 수정] Hard-Switch가 아닌 "Smooth Transition" 적용
            # =====================================================================
            # 1단계 가중치 (W_app만 적용)
            
            # [버그 2 수정] schedule_factor를 사용하여 W_phase1에서 W_phase2로 부드럽게 전환
            # schedule_factor = 0.0 -> W_final = W_phase1
            # schedule_factor = 1.0 -> W_final = W_phase2
            W_final = W_app * W_conditional_damp
            N = W_app.shape[0]
            # 4. Self-similarity 마스킹 (기존과 동일)
            identity = torch.eye(N, device=self.device, dtype=torch.bool)
            W_final = W_final.masked_fill(identity, 0.0)

            # ===============================================================
            # ✅ Sanity Check: Idealized Hard / Easy / False Region Mask
            # ===============================================================
            N = len(spatial_labels_np)
            W_ideal = torch.zeros((N, N), device=self.device)

            # mask 정의
            same_spatial = torch.tensor(spatial_labels_np[:, None] == spatial_labels_np[None, :], device=self.device)
            same_motion  = torch.tensor(motion_labels_np[:, None]  == motion_labels_np[None, :],  device=self.device)

            # False Negative (같은 공간, 같은 동작)
            false_mask = same_spatial & same_motion
            # Hard Negative (같은 공간, 다른 동작)
            hard_mask  = same_spatial & ~same_motion
            # Easy Negative (다른 공간)
            easy_mask  = ~same_spatial & ~same_motion
            motion_mask = ~same_spatial & same_motion
            

            # 이상적 weight 할당
            W_ideal[false_mask] = 0.0
            W_ideal[hard_mask]  = 1.0
            W_ideal[easy_mask]  = 0.5
            W_ideal[motion_mask] = 0.0

            # 자기 자신은 0
            W_ideal.fill_diagonal_(0.0)

            # 비교용 로깅
            diff = torch.abs(W_final - W_ideal).mean().item()
            print(f"[Sanity Check] mean|W_final - ideal| = {diff:.4f}")

            W_final_log = W_final.clone().detach()
            W_final = W_ideal



            # --- 7-E: InfoNCE 손실 계산 (양방향) ---
            # 1. 유사도 행렬 (Logits)
            D = z_video_online.shape[1] // 2
            # 1. z_video_online = concat([vid_app, vid_mot])
            ## motion 가중치 similarity 계산
            # gathered_vid_app, gathered_vid_mot = torch.split(gathered_z_video, gathered_z_video.shape[1] // 2, dim=1)
            # gathered_sen_app, gathered_sen_mot = torch.split(gathered_z_sensor, gathered_z_sensor.shape[1] // 2, dim=1)

            # # 3. 최종 sim 계산 (alpha, beta 가중치)
            # alpha, beta = 0.15, 1.00
            # sim_v2s = beta * (gathered_vid_mot @ gathered_sen_mot.T) + \
            #         alpha * (gathered_vid_app @ gathered_sen_app.T)


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
            # lambda_align = 1.0
            lambda_align = 0.0
            lambda_contrastive = 3.0
            # if self.epoch == 0:
                # first epoch is to syncronization.
                # W_final = torch.zeros_like(W_final)
            

            if self.global_rank == 0:
                W_conditional_damp_cpu = W_conditional_damp.detach().cpu().numpy()

                # --- 이제 shape 맞음 ---
                same_mask = labels_np[:, None] == labels_np[None, :]
                diff_mask = labels_np[:, None] != labels_np[None, :]

                same_mean = W_conditional_damp_cpu[same_mask].mean()
                diff_mean = W_conditional_damp_cpu[diff_mask].mean()

                print(f"[DEBUG] Same-label W_damp mean: {same_mean:.4f} | Diff-label W_damp mean: {diff_mean:.4f}")


                sim_vid_mom_cpu = sim_vid_mom.detach().cpu().numpy()

                same_mean_v = sim_vid_mom_cpu[same_mask].mean()
                diff_mean_v = sim_vid_mom_cpu[diff_mask].mean()

                print(f"[DEBUG] Same-label vid_mom mean: {same_mean_v:.4f} | Diff-label vid_mom mean: {diff_mean_v:.4f}")


                sim_sen_mom_cpu = sim_sen_mom.detach().cpu().numpy()

                same_mean_s = sim_sen_mom_cpu[same_mask].mean()
                diff_mean_s = sim_sen_mom_cpu[diff_mask].mean()

                print(f"[DEBUG] Same-label s_mom mean: {same_mean_s:.4f} | Diff-label s_mom mean: {diff_mean_s:.4f}")

                # --- wandb 시각화 동일 ---
                sorted_idx = np.argsort(labels_np)
                W_sorted = W_conditional_damp_cpu[sorted_idx][:, sorted_idx]
                sorted_labels = labels_np[sorted_idx]

                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(W_sorted, cmap='magma', interpolation='nearest')
                plt.colorbar(im, ax=ax, label='W_damp value')

                boundaries = np.where(np.diff(sorted_labels) != 0)[0]
                for b in boundaries:
                    ax.axhline(b + 0.5, color='white', linewidth=0.8)
                    ax.axvline(b + 0.5, color='white', linewidth=0.8)

                ax.set_title(f"W_damp heatmap | same={same_mean:.3f}, diff={diff_mean:.3f}")
                ax.set_xlabel("Samples (sorted by motion label)")
                ax.set_ylabel("Samples (sorted by motion label)")
                plt.tight_layout()

                if self.logger is not None:  # rank 0에서만 wandb 업로드
                    self.logger.experiment.log({
                        "debug/W_motion_heatmap": wandb.Image(fig),
                        "debug/W_motion_same_mean": same_mean,
                        "debug/W_motion_diff_mean": diff_mean
                    })
                plt.close(fig)

                # --- 추가: 각 similarity 히스토그램 ---
                def plot_similarity_hist(mat, name, color_same, color_diff):
                    same_vals = mat[same_mask].flatten()
                    diff_vals = mat[diff_mask].flatten()
                    plt.figure(figsize=(6, 4))
                    plt.hist(same_vals, bins=50, alpha=0.6, color=color_same, label="same-label")
                    plt.hist(diff_vals, bins=50, alpha=0.6, color=color_diff, label="diff-label")
                    plt.title(f"{name} Similarity Dist.\n(same={same_vals.mean():.3f}, diff={diff_vals.mean():.3f})")
                    plt.xlabel("Similarity"); plt.ylabel("Frequency"); plt.legend(); plt.tight_layout()
                    if self.logger is not None:
                        self.logger.experiment.log({f"debug/{name}_hist": wandb.Image(plt)})
                    plt.close()

                plot_similarity_hist(sim_vid_mom_cpu, "Vid_Motion", "royalblue", "lightcoral")
                plot_similarity_hist(sim_sen_mom_cpu, "Sen_Motion", "seagreen", "orange")
                plot_similarity_hist(sim_stable.cpu(), "Stable_Motion", "mediumpurple", "gray")

                print(f"[DEBUG] Logged W_damp heatmap + similarity histograms to wandb")

            if self.global_rank == 0 and (self.global_step % 1) == 0:
                wandb.log({"train/algin loss": align_loss,
                          "train/contrastive loss": custom_contrastive_loss})
                # self.log("train/contrastive loss", custom_contrastive_loss)
                # ✅ Debugging hook
                class_names= {0: 'Open Door 1',
                1: 'Open Door 2',
                2: 'Close Door 1',
                3: 'Close Door 2',
                4: 'Open Fridge',
                5: 'Close Fridge',
                6: 'Open Dishwasher',
                7: 'Close Dishwasher',
                8: 'Open Drawer 1',
                9: 'Close Drawer 1',
                10: 'Open Drawer 2',
                11: 'Close Drawer 2',
                12: 'Open Drawer 3',
                13: 'Close Drawer 3'},
                class_names_list = [
                    'Open Door 1',
                    'Open Door 2',
                    'Close Door 1',
                    'Close Door 2',
                    'Open Fridge',
                    'Close Fridge',
                    'Open Dishwasher',
                    'Close Dishwasher',
                    'Open Drawer 1',
                    'Close Drawer 1',
                    'Open Drawer 2',
                    'Close Drawer 2',
                    'Open Drawer 3',
                    'Close Drawer 3'
                ]
                # class_names_list = [
                #     'Ktch_B4_Cupboard',
                #     'Ktch_Motion_1',
                #     'Ktch_Motion_2',
                #     'Ktch_T2_Cupboard',
                #     'Ktch_T3_Cupboard',
                #     'None Behavior',
                # ]
                try:
                    self.debug_motion_and_weights_regions_by_composite_class(
                        W_app, W_conditional_damp, W_final,             # [N, N]
                        sim_vid_mom, sim_sen_mom, sim_cross_mom, sim_stable, 
                        motion_damp_factor,  # [N, N]
                        labels_np,
                        spatial_labels_np, motion_labels_np,  # [N]
                        class_names=class_names_list,
                        sim_vid_raw=sim_vid_raw,
                        sim_sen_raw=sim_sen_raw,
                        sim_z_vid_raw=sim_z_vid_raw,
                        sim_z_sen_raw=sim_z_sen_raw,
                        step_tag="debug/W_and_MotionSim_regions",
                        logger=self.logger,
                    )
                except Exception as e:
                    print("debug motion pass", e)

                try:
                    self.debug_contrastive_by_label(
                        logits_v2s=logits_v2s.detach(),
                        logits_s2v=logits_s2v.detach(),
                        W_final=W_final.detach(),
                        labels=spatial_labels_gathered,  # or motion_labels
                        temp=self.hparams.contrastive_temp,
                        step_tag="train/contrastive_label_debug",
                        logger=self.logger,
                    )
                except:
                    print("debug contrastive pass")

                try:
                    self.debug_motion_similarity_matrix(
                        sim_vid_mom=sim_vid_mom,
                        sim_sen_mom=sim_sen_mom,
                        motion_labels_np=labels_np,
                        step_tag="debug/W_motion_classwise",
                        logger=self.logger,
                    )
                except:
                    print("debug motion sim pass")


                self.visualize_classwise_motion_similarity(
                    sim_vid_mom=sim_vid_mom,                # [N, D]
                    sim_sen_mom=sim_sen_mom,                # [N, D]
                    motion_labels=labels_gathered,      # [N]
                    class_names=class_names,
                    logger=self.logger
                )


            # cos_reg = F.mse_loss(v_norm @ v_norm.T, v_mom_norm @ v_mom_norm.T.detach())
            # --- 7-F: 최종 손실 결합 ---
            final_loss = (
                # lambda_align * align_loss +
                lambda_contrastive * custom_contrastive_loss
            )
            # -------------------------------------------------------------
            # 6) 종료
            # -------------------------------------------------------------
        def log_grad(module, name="sensor_motion_encoder"):
            total_grad = 0.0
            count = 0
            for p in module.parameters():
                if p.grad is not None:
                    total_grad += p.grad.detach().abs().mean().item()
                    count += 1
            if count == 0:
                print(f"[{name}] ⚠️ no grad found")
                return 0.0
            mean_grad = total_grad / count
            print(f"[{name}] mean abs grad = {mean_grad:.6f}")
            return mean_grad

        # training_step 끝부분 직전에
        if self.global_step % 3 == 0:  # 로그 주기 조절 가능
            g_sensor_motion = log_grad(self.sensor_motion_encoder, "sensor_motion_encoder")
            self.log("debug/grad/sensor_motion_encoder", g_sensor_motion)
            print("sensor motion emb grad", sensor_motion_emb.requires_grad)

            for name, p in self.sensor_motion_encoder.named_parameters():
                if p.grad is not None:
                    print(name, p.grad.abs().mean().item())
                else:
                    print(name, "⚠️ no grad")

            import seaborn as sns
            if self.logger is not None:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))

                sns.heatmap(W_ideal.cpu(), cmap="coolwarm", ax=ax[0], vmin=0, vmax=1)
                sns.heatmap(W_final_log.detach().cpu(), cmap="coolwarm", ax=ax[1], vmin=0, vmax=1)

                ax[0].set_title("Ideal W")
                ax[1].set_title("Computed W_final")

                plt.tight_layout()
                self.logger.experiment.log({"Sanity_W_Matrix": wandb.Image(fig)})
                plt.close(fig)



            
        if (self.global_step % 400) == 0 and self.global_rank == 0:
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
                print("skip [vis] ", e)
        print("final_loss!! ", final_loss)
        return final_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        매 학습 배치(step)가 끝난 후 호출됩니다.
        training_step의 else 블록과 동일한 조건(freeze_epoch 이후)에서만
        모멘텀 인코더를 업데이트합니다.
        """
        # training_step의 분기문과 동일하게 조건부 실행
        if self.stage2_only:
            print("update momentum")
            self._update_momentum_encoders()


            # epoch 종료 시 한번만 호출됨
    def on_train_epoch_end(self):    
            # 에포크가 끝난 후 epoch 업데이트
        self.eval()  
      # 1️⃣ rank 0이 아닌 프로세스는 먼저 대기 (pre-save barrier)
        if dist.is_initialized() and self.global_rank != 0:
            print(f"[rank{self.global_rank}] waiting before save")
            dist.barrier()

        # 2️⃣ rank 0만 체크포인트 저장
        if self.global_rank == 0:
            save_path = f"/home/jaemo/Method/checkpoints/method/HWU-USP/manual_epochs/manual_epoch_{self.epoch}.ckpt"
            self.trainer.save_checkpoint(save_path)
            print(f"[rank0] checkpoint saved: {save_path}")
            # finally:
            #     # ✅ rank 0이 저장 끝나면 barrier로 나머지 깨워줌
            #     if dist.is_initialized():
            #         dist.barrier()

        # 3️⃣ 나머지 rank도 barrier 통과 후 진행
        if dist.is_initialized() and self.global_rank != 0:
            print(f"[rank{self.global_rank}] resume after rank0 save")
            
        outputs = self.training_steps_outputs
        print(f"{self.global_rank} Epoch {self.epoch} - Collected {len(outputs)} training step outputs.")
        self.epoch += 1
        self.clustering_module.update_epoch(self.epoch)

        with torch.no_grad():
            if self.epoch % self.clustering_module.centroids_update_interval == 0:
                self.clustering_module.clustering_manager.update_centroids_memory()
            
            if self.epoch % self.clustering_module.deal_with_small_clusters_interval == 0:
                self.clustering_module.clustering_manager.deal_with_small_clusters()

        if self.epoch % 2 == 0:
            print(f"\nEpoch {self.epoch}: Running ODC evaluation on training data...")
            self.clustering_module.evaluate(outputs)
        
        try:
            if self.global_rank == 0 and len(self.train_video_accs) > 0:
                mean_acc = torch.stack(self.train_video_accs).mean().item()
                wandb.log({"train/video_classifier_acc_mapped": mean_acc, "epoch": self.epoch})
                print(f"[Epoch {self.epoch}] Video classifier acc (avg): {mean_acc:.4f}")
                self.train_video_accs.clear()
        except Exception as e:
            print(e)
        
        try:
            if self.global_rank == 0 and hasattr(self, "cm_data"):
                preds_list = self.cm_data.get("preds", [])
                labels_list = self.cm_data.get("labels", [])

                if len(preds_list) > 0 and len(labels_list) > 0:
                    preds_all = torch.cat(preds_list).numpy()
                    labels_all = torch.cat(labels_list).numpy()

                    from sklearn.metrics import confusion_matrix
                    import seaborn as sns
                    import matplotlib.pyplot as plt

                    cm = confusion_matrix(labels_all, preds_all)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(f"Confusion Matrix")

                    wandb.log({
                        f"video_classifier/confusion_matrix": wandb.Image(fig)
                    })
                    plt.close(fig)
                else:
                    print(f"[Warn] Skipping confusion matrix log — no predictions in cm_data (len={len(preds_list)})")

                # 초기화
                self.cm_data = {"preds": [], "labels": []}
        except Exception as e:
            print(e)
         
         # ✅ confidence 시각화
        try:
            if self.global_rank == 0 and hasattr(self, "video_confidence_log"):
                import matplotlib.pyplot as plt
                import numpy as np
                import seaborn as sns

                confs = torch.cat(self.video_confidence_log).numpy()
                plt.figure(figsize=(6, 4))
                sns.histplot(confs, bins=20, kde=True, color='steelblue')
                plt.title(f"Video Classifier Confidence")
                plt.xlabel("Max Probability")
                plt.ylabel("Frequency")
                plt.grid(alpha=0.3)

                wandb.log({
                    f"video_classifier/confidence_hist": wandb.Image(plt)
                })
                plt.close()
                self.video_confidence_log = []  # 초기화
            
            if self.global_rank == 0 and hasattr(self, "log_buffer"):
                acc_all = torch.stack(self.log_buffer["classifier_all"]).mean().item()
                acc_good = torch.stack(self.log_buffer["acc_good"]).mean().item()
                acc_bad = torch.stack(self.log_buffer["acc_bad"]).mean().item()
                good_ratio = torch.stack(self.log_buffer["good_ratio"]).mean().item()

                wandb.log({
                    "video_classifier/train_classifier_all": acc_all,
                    "video_classifier/train_acc_good": acc_good,
                    "video_classifier/train_acc_bad": acc_bad,
                    "video_classifier/train_good_ratio": good_ratio,
                    "epoch": self.current_epoch,
                })

                print(f"[Epoch {self.current_epoch}] acc_all={acc_all:.3f}, acc_good={acc_good:.3f}, acc_bad={acc_bad:.3f}, good_ratio={good_ratio:.2f}")

                # 버퍼 초기화
                self.log_buffer = {}
                self.cm_data = {"preds": [], "labels": []}
        except Exception as e:
            print(e)

        print("evaluate called on rank", self.global_rank)
        # 매 2 에폭마다 훈련 데이터셋에 대한 클러스터링 성능 평가


        self.train()  # 모델을 다시 훈련 모드로 설정
        print("Epoch end processing completed on rank", self.global_rank)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
   
    @torch.no_grad()
    def debug_motion_and_weights_regions_by_composite_class(
        self,
        W_app, W_damp, W_final,             # [N, N]
        sim_vid_mom, sim_sen_mom, sim_cross_mom, # [N, N]
        sim_stable, motion_damp_factor,      # [N, N]
        labels_np, spatial_labels_np, motion_labels_np,  # [N], [N], [N]
        class_names,  # 총 14개 라벨 이름 (예: ["Door1-Open", "Door1-Close", ...])
        sim_vid_raw=None,
        sim_sen_raw=None,
        sim_z_vid_raw=None,
        sim_z_sen_raw=None,
        step_tag="debug/W_and_MotionSim_byCompositeClass",
        logger=None,
    ):
        """
        🎯 Class-wise symmetric region analysis based on (spatial, motion) decomposition.
        - labels_np: 전체 클래스 (예: 14개)
        - spatial_labels_np: 7개 공간적 클래스
        - motion_labels_np: 2개 동작 클래스
        """

        # ===============================================================
        # 1️⃣ 기본 세팅
        # ===============================================================
        W_app_np   = W_app.detach().cpu().numpy()
        W_damp_np  = W_damp.detach().cpu().numpy()
        W_final_np = W_final.detach().cpu().numpy()
        motion_damp_np = motion_damp_factor.detach().cpu().numpy()
        sim_vid_np = sim_vid_mom.detach().cpu().numpy()
        sim_sen_np = sim_sen_mom.detach().cpu().numpy()
        sim_cross_np = sim_cross_mom.detach().cpu().numpy()
        sim_stb_np = sim_stable.detach().cpu().numpy()
        sim_vid_raw_np = sim_vid_raw.detach().cpu().numpy() if sim_vid_raw is not None else None  # ✅ (2)
        sim_sen_raw_np = sim_sen_raw.detach().cpu().numpy() if sim_sen_raw is not None else None  # ✅ (2)
        sim_z_vid_raw_np = sim_z_vid_raw.detach().cpu().numpy() if sim_z_vid_raw is not None else None  # ✅ (2)
        sim_z_sen_raw_np = sim_z_sen_raw.detach().cpu().numpy() if sim_z_sen_raw is not None else None  # ✅ (2)

        N = W_app_np.shape[0]
        num_classes = len(class_names)

        mask_upper = np.triu(np.ones((N, N), dtype=bool), k=1)

        # ===============================================================
        # 2️⃣ Pair 관계 정의
        # ===============================================================
        same_spatial = (spatial_labels_np[:, None] == spatial_labels_np[None, :])
        same_motion  = (motion_labels_np[:, None]  == motion_labels_np[None, :])

        hard_mask   = (same_spatial & ~same_motion) & mask_upper
        false_mask  = (same_spatial & same_motion) & mask_upper
        motion_same_mask = (~same_spatial &  same_motion) & mask_upper  # ✅ 다른 spatial, 같은 motion
        motion_diff_mask = (~same_spatial & ~same_motion) & mask_upper  # ✅


        # ===============================================================
        # 3️⃣ 통계 헬퍼
        # ===============================================================
        def compute_stats(vals):
            if vals.size == 0:
                return {k: np.nan for k in ["mean", "median", "min", "max", "q1", "q3", "count"]}
            return {
                "mean": np.mean(vals),
                "median": np.median(vals),
                "min": np.min(vals),
                "max": np.max(vals),
                "q1": np.quantile(vals, 0.25),
                "q3": np.quantile(vals, 0.75),
                "count": vals.size,
            }

        def compute_all_stats(mat, mask):
            out = {
                "W_app": compute_stats(W_app_np[mask]),
                "W_damp": compute_stats(W_damp_np[mask]),
                "W_final": compute_stats(W_final_np[mask]),
                "Stable": compute_stats(sim_stb_np[mask]),
                "Vid_mom": compute_stats(sim_vid_np[mask]),
                "Sen_mom": compute_stats(sim_sen_np[mask]),
                "Cross_Sim": compute_stats(sim_cross_np[mask]),
                "MotDamp": compute_stats(motion_damp_np[mask]),
            }
            if sim_vid_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
                out["Vid_mom_raw"] = compute_stats(sim_vid_raw_np[mask])
            if sim_sen_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
                out["Sen_mom_raw"] = compute_stats(sim_sen_raw_np[mask])
            if sim_z_vid_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
                out["Z_vid_mom_raw"] = compute_stats(sim_z_vid_raw_np[mask])
            if sim_z_sen_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
                out["Z_sen_mom_raw"] = compute_stats(sim_z_sen_raw_np[mask])
            return out

        # ===============================================================
        # 4️⃣ Class-wise 통계 계산
        # ===============================================================
        print(f"\n[{step_tag}] Composite-class (spatial×motion) region stats\n")
        print(f"{'Class':<20} | {'Type':<6} | {'Cnt':<6} | {'W_app':<6} | {'W_dmp':<6} | {'W_fin':<6} | {'Stbl':<6} | {'Vid':<6} | {'Sen':<6} | {'Sen_mom_raw':<6} | {'Z_sen_mom_raw':<6} | {'Cros':<6}")
        #print(f"{'Class':<20} | {'Type':<6} | {'Cnt':<6} | {'W_app':<6} | {'W_dmp':<6} | {'W_fin':<6} | {'Stbl':<6} | {'Vid':<6} | {'Sen':<6} | {'Vid_mom_raw':<6} | {'Sen_mom_raw':<6} | {'Z_vid_mom_raw':<6} | {'Z_sen_mom_raw':<6} | {'Cros':<6}")
        print("-" * 195)

        wandb_data = {}

        for class_idx, class_name in enumerate(class_names):
            # 현재 클래스 샘플 인덱스
            class_mask = (labels_np == class_idx)
            if not np.any(class_mask):
                continue

            # 이 클래스의 샘플들이 포함된 페어
            class_pair_mask = (class_mask[:, None] | class_mask[None, :]) & mask_upper

            # Hard/False 교집합
            hard_pairs = hard_mask & class_pair_mask
            false_pairs = false_mask & class_pair_mask
            # print("mean and median")
            # print("q1 and q3")

            region_dict = {
                "Hard": hard_mask & class_pair_mask,
                "False": false_mask & class_pair_mask,
                "Easy(o)": motion_same_mask & class_pair_mask,
                "Easy(x)": motion_diff_mask & class_pair_mask,
            }

            for region_name, region_mask in region_dict.items():
                stats = compute_all_stats(W_app_np, region_mask)
                count = stats["W_app"]["count"]
                if count == 0:
                    continue

                print(f"{class_name:<20} | {region_name:<6} | {count:<6} | "
                  f"{stats['W_app']['mean']:<6.2f} | "
                  f"{stats['W_damp']['mean']:<6.2f} | "
                  f"{stats['W_final']['mean']:<6.2f} | "
                  f"{stats['Stable']['mean']:<6.2f} | "
                #   f"{stats['Vid_mom']['mean']:<6.2f} | "
                  f"{stats['Sen_mom']['mean']:<6.2f} | "
                #   f"{(stats['Vid_mom_raw']['mean'] if 'Vid_mom_raw' in stats else np.nan):<15.2f} | "
                  f"{(stats['Sen_mom_raw']['mean'] if 'Sen_mom_raw' in stats else np.nan):<15.2f} | "
                #   f"{(stats['Z_vid_mom_raw']['mean'] if 'Z_vid_mom_raw' in stats else np.nan):<15.2f} | "
                  f"{(stats['Z_sen_mom_raw']['mean'] if 'Z_sen_mom_raw' in stats else np.nan):<15.2f} | "
                  f"{stats['Cross_Sim']['mean']:<6.2f}")

            print("--------------------------")
        
        print(f"\n[{step_tag}] Composite-class (spatial×motion) region stats\n")
        print(f"{'Class':<20} | {'Type':<6} | {'Cnt':<6} | {'W_app':<13} | {'W_dmp':<13} | {'W_fin':<13} | {'Stbl':<1e} | {'Vid':<10} | {'Sen':<10} | {'Sen_mom_raw':<6} | {'Cros':<6}")
        #print(f"{'Class':<20} | {'Type':<6} | {'Cnt':<6} | {'W_app':<6} | {'W_dmp':<6} | {'W_fin':<6} | {'Stbl':<6} | {'Vid':<6} | {'Sen':<6} | {'Vid_mom_raw':<6} | {'Sen_mom_raw':<6} | {'Z_vid_mom_raw':<6} | {'Z_sen_mom_raw':<6} | {'Cros':<6}")
        print("-" * 195)

        for class_idx, class_name in enumerate(class_names):
            # 현재 클래스 샘플 인덱스
            class_mask = (labels_np == class_idx)
            if not np.any(class_mask):
                continue

            # 이 클래스의 샘플들이 포함된 페어
            class_pair_mask = (class_mask[:, None] | class_mask[None, :]) & mask_upper

            # Hard/False 교집합
            hard_pairs = hard_mask & class_pair_mask
            false_pairs = false_mask & class_pair_mask
            # print("mean and median")
            # print("q1 and q3")

            for region_name, region_mask in region_dict.items():
                stats = compute_all_stats(W_app_np, region_mask)
                count = stats["W_app"]["count"]
                if count == 0:
                    continue    # 바로 밑줄에 Q1~Q3 표시 추가

                print(f"{class_name:<20} | {region_name:<6} | {count:<6}  | "
                    f"{stats['W_app']['q1']:<6.2f}/{stats['W_app']['q3']:<6.2f} | "
                    f"{stats['W_damp']['q1']:<6.2f}/{stats['W_damp']['q3']:<6.2f} | "
                    f"{stats['W_final']['q1']:<6.2f}/{stats['W_final']['q3']:<6.2f} | "
                    f"{stats['Stable']['q1']:<6.2f}/{stats['Stable']['q3']:<6.2f} | "
                    f"{stats['Vid_mom']['q1']:<6.2f}/{stats['Vid_mom']['q3']:<6.2f} | "
                    f"{stats['Sen_mom']['q1']:<6.2f}/{stats['Sen_mom']['q3']:<6.2f} | "
                    # f"{(stats['Vid_mom_raw']['q1'] if 'Vid_mom_raw' in stats else np.nan):<15.2f} | "
                    # f"{(stats['Sen_mom_raw']['q1'] if 1'Sen_mom_raw' in stats else np.nan):<15.2f} | "
                    # f"{(stats['Z_vid_mom_raw']['q1'] if 'Z_vid_mom_raw' in stats else np.nan):<15.2f} | "
                    # f"{(stats['Z_sen_mom_raw']['q1'] if 'Z_sen_mom_raw' in stats else np.nan):<15.2f} | "
                    f"{stats['Cross_Sim']['q1']:<6.2f}/{stats['Cross_Sim']['q3']:<6.2f}")

                # wandb 로깅용
                for key, s in stats.items():
                    for stat_key in ["mean", "median", "count"]:
                        wandb_data[f"{step_tag}/{class_name}/{region_name}/{key}_{stat_key}"] = s[stat_key]
            print("----------------")

        # ===============================================================
        # 5️⃣ WandB 로그 업로드
        # ===============================================================
        if logger is not None and len(wandb_data) > 0:
            logger.experiment.log(wandb_data)

        print(f"\n✅ [{step_tag}] composite class-based symmetric region stats computed.")


    @torch.no_grad()
    def visualize_classwise_motion_similarity(
        self,
        sim_vid_mom, sim_sen_mom,  # [N, N] Pre-computed similarity matrices
        motion_labels,             # [N]
        class_names=None,
        logger=None
    ):
        """
        현재 batch (또는 전체 데이터)에서 등장한 클래스 n개에 대해:
        - (i, j): 클래스 i와 j의 평균 cosine similarity
        - i==j 인 경우 자기 자신 제외 (intra-class mean)
        - 입력으로 이미 계산된 N*N 유사도 행렬을 받아서 효율적으로 처리
        """
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import wandb

        # ---------------------------------------------------------------------
        #  PREPARE LABELS
        # ---------------------------------------------------------------------
        labels_np = motion_labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        num_classes = len(unique_labels)
        
        # 클래스 이름 매핑 (없으면 숫자 그대로 사용)
        if class_names:
            # class_names가 dict인 경우와 list인 경우 모두 처리
            if isinstance(class_names, dict):
                 class_names_used = [class_names.get(i, str(i)) for i in unique_labels]
            else:
                 class_names_used = [class_names[i] if i < len(class_names) else str(i) for i in unique_labels]
        else:
            class_names_used = [str(i) for i in unique_labels]

        # ---------------------------------------------------------------------
        #  CLASSWISE SIMILARITY COMPUTATION (Optimized)
        # ---------------------------------------------------------------------
        def compute_classwise_from_sim_matrix(sim_matrix_tensor, labels_np, unique_labels):
            n = len(unique_labels)
            class_sim_matrix = np.zeros((n, n))
            
            # 텐서를 CPU numpy로 변환 (인덱싱 편의를 위해)
            sim_np = sim_matrix_tensor.detach().cpu().numpy()

            for i, ci in enumerate(unique_labels):
                idx_i = np.where(labels_np == ci)[0]
                if len(idx_i) == 0: continue

                for j, cj in enumerate(unique_labels):
                    idx_j = np.where(labels_np == cj)[0]
                    if len(idx_j) == 0: continue

                    # ✅ 이미 계산된 유사도 행렬에서 해당 클래스 쌍의 부분 행렬 추출
                    # sim_np[idx_i][:, idx_j] -> [Ni, Nj] 형태의 부분 행렬
                    # (ixgrid를 사용하여 효율적으로 추출)
                    sub_sim = sim_np[np.ix_(idx_i, idx_j)]

                    if ci == cj:
                        # 🔹 Intra-class: 자기 자신과의 비교(대각선 성분) 제외
                        if len(idx_i) > 1:
                            # 대각선 마스크 생성 (Ni x Ni)
                            mask = ~np.eye(len(idx_i), dtype=bool)
                            class_sim_matrix[i, j] = sub_sim[mask].mean()
                        else:
                            # 샘플이 1개뿐이면 자기 자신과의 유사도(1.0)를 제외할 수 없음 -> NaN 또는 1.0 처리
                            # 여기서는 의미상 NaN이 맞으나, 편의상 1.0으로 둘 수도 있음. (보통 이런 경우는 드묾)
                            class_sim_matrix[i, j] = np.nan 
                    else:
                        # 🔹 Inter-class: 모든 쌍의 평균
                        class_sim_matrix[i, j] = sub_sim.mean()

            return class_sim_matrix

        # ✅ Compute using pre-calculated similarity matrices
        sim_v_class = compute_classwise_from_sim_matrix(sim_vid_mom, labels_np, unique_labels)
        sim_s_class = compute_classwise_from_sim_matrix(sim_sen_mom, labels_np, unique_labels)

        # ---------------------------------------------------------------------
        #  PLOT HEATMAP
        # ---------------------------------------------------------------------
        def plot_heatmap(mat, title, modality):
            # NaN 값이 있을 경우 처리 (예: 0으로 대체하거나 마스킹)
            mat_safe = np.nan_to_num(mat, nan=0.0)
            
            fig, ax = plt.subplots(figsize=(10, 8)) # 크기 약간 키움
            sns.heatmap(mat_safe, cmap="magma", vmin=0, vmax=1,
                        xticklabels=class_names_used, yticklabels=class_names_used,
                        ax=ax, annot=True, fmt=".2f", square=True, # 정사각형 모양 유지
                        cbar_kws={"shrink": 0.8}) # 컬러바 크기 조절
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_xlabel("Class j", fontsize=12)
            ax.set_ylabel("Class i", fontsize=12)
            plt.xticks(rotation=45, ha="right") # X축 라벨 회전
            plt.yticks(rotation=0)
            plt.tight_layout()

            if logger:
                logger.experiment.log({f"debug/classwise_sim/{modality}": wandb.Image(fig)})
            plt.close(fig)

        plot_heatmap(sim_v_class, "Class-wise Motion Similarity (Video)", "video")
        plot_heatmap(sim_s_class, "Class-wise Motion Similarity (Sensor)", "sensor")

        # ---------------------------------------------------------------------
        #  CONSOLE SUMMARY (Optional, too large matrices might clutter console)
        # ---------------------------------------------------------------------
        # print(f"[Video Motion] classwise similarity matrix:\n{np.array2string(sim_v_class, precision=2, suppress_small=True)}")