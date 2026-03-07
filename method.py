import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt    
import seaborn as sns
import imageio
import math
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from method_utils import gather, CovarianceAlignmentLoss, overlay_motion_heatmap
import copy
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.cm as cm # Colormap 사용
import matplotlib.colors # Matplotlib Colormap을 numpy 배열로 변환


# --- 사용자 정의 모듈 임포트 ---
from model import SensorEncoder, SensorMotionEncoder, SensorTransformerEncoder, SensorModel, VisionModel, ClusteringModule
from analysis.visualizes import visualize_Wfinal_differences

####################################################################



class MethodLightningModule(pl.LightningModule):

    def __init__(self, args, datamodule=None):
        super().__init__()
        self.save_hyperparameters(args)
        # 1. 모델 구성 요소 초기화

        self.video_model = VisionModel(
            latent_dim=self.hparams.embedding_dim,
            use_flow=self.hparams.use_flow  # Pass use_flow to VisionModel
        )
        self.sensor_appearance_encoder= SensorEncoder(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.sensor_motion_encoder = SensorMotionEncoder(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
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

        if self.hparams.dataset_name == "Opportunity++":
            self.class_names= {0: 'Open Door 1',
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
                    13: 'Close Drawer 3'}
        else:            
            self.class_names = {
                0: 'Ktch_B4_Cupboard',
                1: 'Ktch_Motion_1',
                2: 'Ktch_Motion_2',
                3: 'Ktch_T2_Cupboard',
                4: 'Ktch_T3_Cupboard',
                5: 'None Behavior',
            }

        self.class_names_list = list(self.class_names.values())
        
        # ============================================================
        # Parameter Count Summary
        # ============================================================
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        # Stage1 사용 모듈 (motion encoder 제외)
        stage1_modules = {
            "video_model (appearance only)": self.video_model,
            "sensor_appearance_encoder": self.sensor_appearance_encoder,
            "clustering_module": self.clustering_module,
            "video_classifier": self.video_classifier,
            "appearance_classifier": self.appearance_classifier,
        }
        
        # Stage2 추가 모듈
        stage2_additional = {
            "sensor_motion_encoder": self.sensor_motion_encoder,
        }
        
        stage1_total = sum(count_params(m) for m in stage1_modules.values())
        stage2_additional_total = sum(count_params(m) for m in stage2_additional.values())
        
        print("=" * 60)
        print("Method Model Parameter Count")
        print("=" * 60)
        print(f"[Stage1] Total: {stage1_total:,} ({stage1_total/1e6:.2f}M)")
        for name, m in stage1_modules.items():
            print(f"  {name:35s}: {count_params(m):>12,}")
        print("-" * 60)
        print(f"[Stage2 Additional]: {stage2_additional_total:,} ({stage2_additional_total/1e6:.2f}M)")
        for name, m in stage2_additional.items():
            print(f"  {name:35s}: {count_params(m):>12,}")
        print("-" * 60)
        print(f"[Total (Stage1+Stage2)]: {stage1_total + stage2_additional_total:,} ({(stage1_total + stage2_additional_total)/1e6:.2f}M)")
        print("=" * 60)


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
        if self.hparams.ablation_study is not None:
            cache_dir = os.path.join(self.hparams.cache_dir, f"ablations/{self.hparams.ablation_study}")
        else:
            cache_dir = self.hparams.cache_dir
        # 첫 에폭에서 메모리 뱅크 초기화 (중요!)
        if self.epoch == 0:
  
            print("Initializing memory bank at epoch 0...")
            self.clustering_module.init_prototypes_with_data(self.device, self.hparams.num_classes)
            self.stage2_only = False
            if self.hparams.threshold_epoch == -1:
                ckpt_path = os.path.join(cache_dir, f"threshold_epoch=5.pt")
                state = torch.load(ckpt_path, map_location="cpu")
                self.clustering_module.load_state_dict(state["cluster_model"])
                self.clustering_module.clustering_manager.feature_bank = state["memory"].to(self.device)
                self.clustering_module.clustering_manager.centroids = state["centroids"].to(self.device)
                self.clustering_module.clustering_manager.label_bank = state["label_bank"].to(self.device)

                # ⚠️ 중요: buffer까지 GPU로 강제 이동
                self.clustering_module.to(self.device)

                print(f"[✔] Loaded pretrained clustering model from {ckpt_path}")

            elif self.hparams.threshold_epoch == -2 and self.hparams.video_classifier_epoch == 2:
                ckpt_dir = cache_dir
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
                ckpt_dir = cache_dir
                ckpt_candidates = [
                    os.path.join(ckpt_dir, "start_stage2_epoch=25.pt"),
                    os.path.join(ckpt_dir, "start_stage2_epoch=20.pt"),
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
                    ckpt_path = os.path.join(cache_dir, f"threshold_epoch={self.hparams.threshold_epoch}.pt")
                    torch.save({
                        "cluster_model": self.clustering_module.state_dict(),
                        "memory": self.clustering_module.clustering_manager.feature_bank,
                        "centroids": self.clustering_module.clustering_manager.centroids,
                        "label_bank": self.clustering_module.clustering_manager.label_bank,
                    }, ckpt_path)

                    print(f"[✔] Saved clustering stage-1 weights → {ckpt_path}")
                
                elif self.epoch == self.hparams.threshold_epoch + self.hparams.video_classifier_epoch and self.global_rank == 0:
                    classifier_ckpt_path = os.path.join(cache_dir, f"vision_model_classifier_cetroid_threshold={self.hparams.centroid_threshold}_epoch={self.epoch}.pt")
                    torch.save({
                        "cluster_model": self.clustering_module.state_dict(),
                        "memory": self.clustering_module.clustering_manager.feature_bank,
                        "centroids": self.clustering_module.clustering_manager.centroids,
                        "label_bank": self.clustering_module.clustering_manager.label_bank,
                        "vision_model": self.video_model.state_dict()},
                    classifier_ckpt_path)
                    print(f"[✔] Saved VisionModel weights → {classifier_ckpt_path}")

                
                elif self.epoch == self.hparams.threshold_epoch + self.hparams.video_classifier_epoch + self.hparams.bad_correction_epoch:
                    stage1_ckpt_path = os.path.join(cache_dir, f"start_stage2_epoch={self.epoch}.pt")
                    torch.save({
                        "cluster_model": self.clustering_module.state_dict(),
                        "memory": self.clustering_module.clustering_manager.feature_bank,
                        "centroids": self.clustering_module.clustering_manager.centroids,
                        "label_bank": self.clustering_module.clustering_manager.label_bank,
                        "vision_model": self.video_model.state_dict()},
                    stage1_ckpt_path)
                    print(f"[✔] Saved VisionModel weights → {stage1_ckpt_path}")
        self.training_steps_outputs = []  # 에포크 동안의 출력 저장용
        self.debug = []
        stage2_start = (
            self.hparams.threshold_epoch +
            self.hparams.video_classifier_epoch +
            self.hparams.bad_correction_epoch
        )
        if self.epoch >= stage2_start: 
            if not self.stage2_only:
                self.enable_stage2()


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
                    if self.hparams.centroid_threshold == 0.75:
                        threshold = manager.cluster_stats[k]["q75"]
                    elif self.hparams.centroid_threshold == 0.50:
                        threshold = manager.cluster_stats[k]["q50"]
                    elif self.hparams.centroid_threshold == 1.00:
                        threshold = manager.cluster_stats[k]["all"]
                    else:
                        raise("not calculated stat")
                    good_mask[cluster_mask] = cluster_dists < threshold

            if self.hparams.centroid_threshold == 1.00:
                # ablation
                bad_mask = good_mask
            else:
                bad_mask = ~good_mask
            bad_indicator = bad_mask.long()

            # --- 거리 정보 저장 ---
            distance_info = {
                'centroid_distances': centroid_distances.cpu(),
                'pseudo_labels': stored_pseudo_labels.cpu(),
            }

            # 메모리 업데이트
            change_ratio = self.clustering_module.clustering_manager.update_samples_memory(idx, features)
            
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
                'bad': bad_indicator.detach(),
                'change_ratio': change_ratio.detach()
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

        # --- 최종 손실 계산 ---
        lambda_cluster = 1.0
        lambda_video_sup = 1.0
        lambda_sensor_guide = 1.5

        final_loss = (
            lambda_cluster * loss_cluster +
            lambda_video_sup * loss_video_supervised +
            lambda_sensor_guide * loss_sensor_guided # no refinement
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
        if self.epoch > stage2_start: 

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
            gathered_v_app = F.normalize(gather(v_appearance.detach()), dim=1)
            gathered_features = F.normalize(gather(features.detach()), dim=1)
            z_sensor = gather(z_sensor_online)
            
            self.training_steps_outputs[-1]["z_video"] = z_video_online
            # self.training_steps_outputs[-1]["z_video"] = F.normalize((z_video_online), dim=1)
            self.training_steps_outputs[-1]["z_sensor"] = F.normalize((z_sensor_online), dim=1)
            self.training_steps_outputs[-1]["v_motion"] = F.normalize((v_motion), dim=1)
            self.training_steps_outputs[-1]["s_motion"]= F.normalize((sensor_motion_emb), dim=1)
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
            spatial_labels_gathered = gather(spatial_labels)
            spatial_labels_np = spatial_labels_gathered.cpu().numpy()

              # --- 7-D: 가중치 행렬 계산 (W = W_app * W_motion_damp) ---
            # 1. Appearance 비중 조절 (온건한 re-weighting)
            sim_app_vid = gathered_v_app @ gathered_v_app.T
            sim_app_sen = gathered_features @ gathered_features.T
            # [수정됨] F.relu()를 torch.max() 밖으로 이동하여 음수 유사도를 먼저 처리
            sim_app_vid_rel = F.relu(sim_app_vid)
            sim_app_sen_rel = F.relu(sim_app_sen)
            sim_app_max = torch.max(sim_app_vid_rel, sim_app_sen_rel) # [N, N]
            # [버그 1 수정] sim_app_avg -> sim_app_max
            W_app = 1.0 + (self.hparams.lambda_hard - 1.0) * sim_app_max
            # 2. Motion 감쇠 (FN filtering)
            sim_vid_mom = gathered_v_mom @ gathered_v_mom.T
            sim_sen_mom = gathered_s_mom @ gathered_s_mom.T
            sim_cross_mom = gathered_v_mom @ gathered_s_mom.T
            
            # ------------------------------------------------------------------
            # ❗중요❗: 이 후에 꼭 L2 Normalization을 해줘야 코사인 유사도가 됩니다.
            # Centering 후 L2 Norm = "평균으로부터의 각도(방향) 차이"
            # ------------------------------------------------------------------
            # v_motion_norm = F.normalize(v_motion_norm, dim=1)
            # s_motion_norm = F.normalize(s_motion_norm, dim=1)
            
            # sim_vid_mom = v_motion_norm @ v_motion_norm.T
            # sim_sen_mom = s_motion_norm @ s_motion_norm.T
            # sim_cross_mom = s_motion_norm @ v_motion_norm.T
            sim_cross = gathered_v_mom @ gathered_s_mom.T
            sim_stable = sim_cross

         
            
            motion_damp_factor = F.relu(sim_stable)
            # motion_damp_factor = torch.tanh(sim_stable / 0.07)
            # [버그 1 수정] sim_app_avg -> sim_app_max
            W_conditional_damp = 1.0 - sim_app_max * motion_damp_factor
            
            W_final = W_app * W_conditional_damp

            # ✅ Ablation Study: W_final 1.0으로 고정 (단순 InfoNCE)
            # -------------------------------------------------------------
            
            # W_final을 1.0으로 채워진 텐서로 생성 (W_app과 동일한 shape/device)
            # W_final = torch.ones_like(W_app)

            # Ablation Study: W_app 사용 x
            # W_final = W_conditional_damp
            # Ablation Study: W_cond 사용 x
            # W_final = W_app
            N = W_app.shape[0]
            # 4. Self-similarity 마스킹 (기존과 동일)
            identity = torch.eye(N, device=self.device, dtype=torch.bool)
            W_final = W_final.masked_fill(identity, 0.0)

            if self.epoch == 0:
                W_final = torch.zeros_like(W_app)

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
            W_ideal[hard_mask]  = 2.0
            W_ideal[easy_mask]  = 0.5
            W_ideal[motion_mask] = 0.1

            # 자기 자신은 0
            W_ideal.fill_diagonal_(0.0)

            # 비교용 로깅
            diff = torch.abs(W_final - W_ideal).mean().item()
            print(f"[Sanity Check] mean|W_final - ideal| = {diff:.4f}")

            W_final_log = W_final.clone().detach()
            # W_final = W_ideal
            # W_final = W_final.detach()
            # ===============================================================



            # --- 7-E: InfoNCE 손실 계산 (양방향) ---
            # 1. 유사도 행렬 (Logits)
            D = z_video_online.shape[1] // 2

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
            # 3. 가중치가 적용된 네거티브 합 계산x
            # W_final은 대칭이므로 (W_final.T == W_final) 동일하게 사용
            weighted_neg_v2s = (W_final * torch.exp(neg_logits_v2s)).sum(dim=1)
            weighted_neg_s2v = (W_final * torch.exp(neg_logits_s2v)).sum(dim=1)
            # 4. 양방향 손실 계산
            denominator_v2s = torch.exp(pos_v2s) + weighted_neg_v2s
            denominator_s2v = torch.exp(pos_s2v) + weighted_neg_s2v
            loss_v2s = -torch.log(torch.exp(pos_v2s) / denominator_v2s).mean()
            loss_s2v = -torch.log(torch.exp(pos_s2v) / denominator_s2v).mean()
            custom_contrastive_loss = (loss_v2s + loss_s2v) / 2.0
            # lambda_align = 4.0
            lambda_contrastive = 1.0

            self.debug.append({
                # --- (Stage 2 디버깅용) ---
                'W_app': W_app.detach(),
                'W_damp': W_conditional_damp.detach(),
                'W_final': W_final.detach(),
                'sim_vid_mom': sim_vid_mom.detach(),
                'sim_sen_mom': sim_sen_mom.detach(),
                'sim_cross_mom': sim_cross_mom.detach(),
                'sim_stable': sim_stable.detach(),
                'motion_damp_factor': motion_damp_factor.detach(),
                'sim_app_vid': sim_app_vid.detach() if sim_app_vid is not None else None,
                'sim_app_sen': sim_app_sen.detach() if sim_app_sen is not None else None,
                'spatial_labels_np': spatial_labels_np, # 이미 numpy
                'motion_labels_np': motion_labels_np,   # 이미 numpy
                'labels_np': gather(labels).cpu().numpy(), # GT 라벨
            })

            def _safe_detach(x):
                return x.detach().cpu()

            # self.debug.append({
            #     'z_vid_mom': _safe_detach(sim_vid_mom),   # mean X
            #     'z_sen_mom': _safe_detach(sim_sen_mom),
            #     'labels_np': gather(labels).cpu().numpy(),
            #     'spatial_labels_np': spatial_labels_np,
            #     'motion_labels_np': motion_labels_np,
            # })

            

            # if self.global_rank == 0 and (batch_idx % 5 == 0):
            #     print(type(labels[0]))
            #     if self.hparams.dataset_name=="Opportunity++" or labels[0] in [0, 3, 4, 5]:
                        
            #         # ======================================================
            #         # I. 데이터 준비
            #         # ======================================================
            #         raw_feature_map = model_output["motion_feature_map"]   # [B, C, T', H', W']
            #         single_motion_feat = raw_feature_map[0].detach().cpu() # [C, T', H', W']

            #         _, T_orig, _, H_orig, W_orig = videos.shape
                    
            #         # Appearance shared feature map
            #         spatial_feat = model_output["spatial_feature_map"][0].detach().cpu()  # [C, H', W']

            #         # =========================================
            #         # II. Appearance Feature Heatmap 준비
            #         # =========================================
            #         # (A) 채널 평균
            #         appearance_map = spatial_feat.mean(dim=0).numpy()    # [H', W']

            #         # (B) Normalize
            #         min_a, max_a = appearance_map.min(), appearance_map.max()
            #         if max_a == min_a:
            #             appearance_norm = np.zeros_like(appearance_map)
            #         else:
            #             appearance_norm = (appearance_map - min_a) / (max_a - min_a + 1e-8)

            #         # (C) Colormap → torch → upsample
            #         cmap = cm.get_cmap('jet')
            #         appearance_rgb = cmap(appearance_norm)[:, :, :3]      # [H',W',3]
            #         appearance_tensor = torch.from_numpy(appearance_rgb).permute(2,0,1).unsqueeze(0).float()

            #         upsampled_appearance = F.interpolate(
            #             appearance_tensor, size=(H_orig, W_orig), mode='bilinear', align_corners=False
            #         )[0].permute(1,2,0).numpy()   # [H,W,3]

            #         # =========================================
            #         # III. 전체 프레임 Motion + Appearance overlay
            #         # =========================================
            #         motion_log_images = []
            #         motion_overlaid_images = []
            #         appearance_overlaid_images = []

            #         C_feat, T_feat, H_feat, W_feat = single_motion_feat.shape

            #         # motion feature map을 frame별로 쓰기 위해 time min(T_feat, T_orig) 맞추기
            #         T_use = min(T_feat, T_orig)

            #         for t in range(T_use):

            #             # ----------------------------------------------------
            #             # 1️⃣ Motion Feature Heatmap (per-frame)
            #             # ----------------------------------------------------
            #             img_map = single_motion_feat[:, t].mean(dim=0).numpy()   # [H',W']

            #             m_min, m_max = img_map.min(), img_map.max()
            #             if m_max == m_min:
            #                 m_norm = np.zeros_like(img_map)
            #             else:
            #                 m_norm = (img_map - m_min) / (m_max - m_min + 1e-8)

            #             # 저장용 raw grayscale
            #             motion_log_images.append(
            #                 wandb.Image((m_norm * 255).astype(np.uint8),
            #                             caption=f"E{self.current_epoch}/{sample_id[0]}_Motion_T{t}")
            #             )

            #             # Jet colormap
            #             motion_rgb = cmap(m_norm)[:, :, :3]   # [H',W',3]
            #             motion_tensor = torch.from_numpy(motion_rgb).permute(2,0,1).unsqueeze(0).float()

            #             motion_up = F.interpolate(
            #                 motion_tensor, size=(H_orig, W_orig),
            #                 mode='bilinear', align_corners=False
            #             )[0].permute(1,2,0).numpy()

            #             # ----------------------------------------------------
            #             # 2️⃣ 원본 프레임 준비 (0~1 float)
            #             # ----------------------------------------------------
            #             frame = videos[0, t].detach().cpu()  # [C,H,W], -1~1이면 아래 필요
            #             frame = frame.mul_(255.0)
            #             frame = frame.clamp(0,1)
            #             frame_np = frame.permute(1,2,0).numpy()

            #             # ----------------------------------------------------
            #             # 3️⃣ Motion Overlay
            #             # ----------------------------------------------------
            #             alpha = 0.5
            #             motion_overlay = (1-alpha)*frame_np + alpha*motion_up
            #             motion_overlay_uint8 = (np.clip(motion_overlay,0,1)*255).astype(np.uint8)

            #             motion_overlaid_images.append(
            #                 wandb.Image(motion_overlay_uint8,
            #                             caption=f"E{self.current_epoch}/{sample_id[0]}_MotionOverlay_T{t}")
            #             )

            #             # ----------------------------------------------------
            #             # 4️⃣ Appearance Overlay (모든 프레임 동일 appearance heatmap)
            #             # ----------------------------------------------------
            #             appearance_overlay = (1-alpha)*frame_np + alpha*upsampled_appearance
            #             appearance_overlay_uint8 = (np.clip(appearance_overlay,0,1)*255).astype(np.uint8)

            #             appearance_overlaid_images.append(
            #                 wandb.Image(appearance_overlay_uint8,
            #                             caption=f"E{self.current_epoch}/{sample_id[0]}_AppearanceOverlay_T{t}")
            #             )

            #         # ======================================================
            #         # WandB Logging
            #         # ======================================================
            #         if self.logger and hasattr(self.logger.experiment, 'log'):
            #             self.logger.experiment.log({
            #                 "motion_feature_maps/raw_sequence": motion_log_images,
            #                 "motion_feature_maps/overlaid_sequence": motion_overlaid_images,
            #                 "appearance_maps/overlaid_sequence": appearance_overlaid_images,
            #                 "epoch": self.current_epoch,
            #                 "global_step": self.global_step,
            #             })


            # --- 7-F: 최종 손실 결합 ---
            final_loss = (
                # lambda_align * align_loss +
                lambda_contrastive * custom_contrastive_loss
            )

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

        # Stage 2가 시작되었고, rank 0일 때만 실행
        if self.stage2_only and self.global_rank == 0:
            print(f"\n[Epoch {self.epoch}] Running Stage-2 Debug Logging...")

            try:
                outputs = self.debug
                if self.hparams.save_weights:
                    save_dir = "/mnt/hdd4tb/junho/Opportunity++/normed_weights"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"epoch_{self.epoch:03d}_stepwise.pkl")

                    # ✅ 각 step별 구조 그대로 보존
                    stepwise_data = []
                    for step_debug in self.debug:
                        step_dict = {
                            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                            for k, v in step_debug.items()
                        }
                        stepwise_data.append(step_dict)

                    # ✅ pickle로 저장 (리스트 구조 그대로 유지)
                    with open(save_path, "wb") as f:
                        pickle.dump(stepwise_data, f)

                    print(f"✅ Saved stepwise debug data ({len(stepwise_data)} steps) → {save_path}")

                    # 메모리 초기화
                self.debug.clear()

            except Exception as e:
                print(f"[ERROR] Failed Epoch Debug Logging: {e}")
            finally:
                self.debug.clear()

        save_epochs = [0, 1, 3, 5, 10, 15, 20, 25]
        if self.epoch in save_epochs or self.epoch % 10 == 0:
            save_path = f"/home/jaemo/Method/checkpoints/method/{self.hparams.dataset_name}/manual_epochs/manual_epoch_{self.epoch}.ckpt"
            self.trainer.save_checkpoint(save_path)
            print(f"[rank0] checkpoint saved: {save_path}")
            
        outputs = self.training_steps_outputs
        print(f"{self.global_rank} Epoch {self.epoch} - Collected {len(outputs)} training step outputs.")
        all_change_ratios = []
        for output in outputs:
            if "change_ratio" in output:
                # change_ratio가 0-dim tensor 또는 float라고 가정하고 리스트에 추가
                all_change_ratios.append(output["change_ratio"])

        if all_change_ratios and self.global_rank == 0:
            # 리스트를 텐서로 변환하여 평균 계산 (모든 change_ratio가 float나 0-dim tensor여야 함)
            avg_change_ratio = torch.mean(torch.tensor(all_change_ratios, device=self.device).float())
            
            # TensorBoard에 로깅
            if self.logger is not None:  # rank 0에서만 wandb 업로드
                self.logger.experiment.log({
                    "epoch/memory_update_ratio":avg_change_ratio,
                    "epoch/epoch": self.current_epoch
                })
            print(f"[✔] Epoch {self.current_epoch} Average Memory Update Ratio: {avg_change_ratio.item():.4f}")

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

        print("evaluate called on rank", self.global_rank)


        self.train()  # 모델을 다시 훈련 모드로 설정
        print("Epoch end processing completed on rank", self.global_rank)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        return optimizer
   
    # @torch.no_grad()
    # def debug_motion_and_weights_regions_by_composite_class(
    #     self,
    #     W_app, W_damp, W_final,             # [N, N]
    #     sim_vid_mom, sim_sen_mom, sim_cross_mom, # [N, N]
    #     sim_stable, motion_damp_factor,      # [N, N]
    #     labels_np, spatial_labels_np, motion_labels_np,  # [N], [N], [N]
    #     class_names,  # 총 14개 라벨 이름 (예: ["Door1-Open", "Door1-Close", ...])
    #     sim_vid_raw=None,
    #     sim_sen_raw=None,
    #     sim_z_vid_raw=None,
    #     sim_z_sen_raw=None,
    #     sim_app_vid=None,
    #     sim_app_sen=None,
    #     step_tag="debug/W_and_MotionSim_byCompositeClass",
    #     logger=None,
    # ):
    #     """
    #     🎯 Class-wise symmetric region analysis based on (spatial, motion) decomposition.
    #     - labels_np: 전체 클래스 (예: 14개)
    #     - spatial_labels_np: 7개 공간적 클래스
    #     - motion_labels_np: 2개 동작 클래스
    #     """

    #     # ===============================================================
    #     # 1️⃣ 기본 세팅
    #     # ===============================================================
    #     W_app_np   = W_app.detach().cpu().numpy()
    #     W_damp_np  = W_damp.detach().cpu().numpy()
    #     W_final_np = W_final.detach().cpu().numpy()
    #     motion_damp_np = motion_damp_factor.detach().cpu().numpy()
    #     sim_vid_np = sim_vid_mom.detach().cpu().numpy()
    #     sim_sen_np = sim_sen_mom.detach().cpu().numpy()
    #     sim_cross_np = sim_cross_mom.detach().cpu().numpy()
    #     sim_stb_np = sim_stable.detach().cpu().numpy()
    #     sim_vid_raw_np = sim_vid_raw.detach().cpu().numpy() if sim_vid_raw is not None else None  # ✅ (2)
    #     sim_sen_raw_np = sim_sen_raw.detach().cpu().numpy() if sim_sen_raw is not None else None  # ✅ (2)
    #     sim_z_vid_raw_np = sim_z_vid_raw.detach().cpu().numpy() if sim_z_vid_raw is not None else None  # ✅ (2)
    #     sim_z_sen_raw_np = sim_z_sen_raw.detach().cpu().numpy() if sim_z_sen_raw is not None else None  # ✅ (2)
        
    #     sim_app_vid_np = sim_app_vid.detach().cpu().numpy() if sim_app_vid is not None else None  # ✅ (2)
    #     sim_app_sen_np = sim_app_sen.detach().cpu().numpy() if sim_app_sen is not None else None  # ✅ (2)

    #     N = W_app_np.shape[0]
    #     num_classes = len(class_names)

    #     mask_upper = np.triu(np.ones((N, N), dtype=bool), k=1)

    #     # ===============================================================
    #     # 2️⃣ Pair 관계 정의
    #     # ===============================================================
    #     same_spatial = (spatial_labels_np[:, None] == spatial_labels_np[None, :])
    #     same_motion  = (motion_labels_np[:, None]  == motion_labels_np[None, :])

    #     hard_mask   = (same_spatial & ~same_motion) & mask_upper
    #     false_mask  = (same_spatial & same_motion) & mask_upper
    #     motion_same_mask = (~same_spatial &  same_motion) & mask_upper  # ✅ 다른 spatial, 같은 motion
    #     motion_diff_mask = (~same_spatial & ~same_motion) & mask_upper  # ✅

    #             # ===============================================================
    #     # 🧮 6️⃣ Hard vs False classification accuracy (per class)
    #     # ===============================================================
    #     print("\n[Hard-vs-False Accuracy based on W_final ranking]\n")
    #     acc_per_class = {}
    #     auc_per_class = {}

    #     for class_idx, class_name in enumerate(class_names):
    #         # 이 클래스에 해당하는 샘플만 anchor로 선택
    #         class_mask = (labels_np == class_idx)
    #         if not np.any(class_mask):
    #             continue

    #         # Anchor가 이 클래스인 경우의 pair
    #         sub_false = W_final_np[np.ix_(class_mask, np.ones_like(class_mask, dtype=bool))][false_mask[np.ix_(class_mask, np.ones_like(class_mask, dtype=bool))]]
    #         sub_hard  = W_final_np[np.ix_(class_mask, np.ones_like(class_mask, dtype=bool))][hard_mask[np.ix_(class_mask, np.ones_like(class_mask, dtype=bool))]]

    #         if len(sub_false) == 0 or len(sub_hard) == 0:
    #             continue

    #         # 두 분포를 합치고 False는 label=1, Hard는 label=0 으로 표시
    #         vals = np.concatenate([sub_false, sub_hard])
    #         labels = np.concatenate([np.ones_like(sub_false), np.zeros_like(sub_hard)])

    #         # W_final 기준으로 오름차순 정렬 (낮을수록 "False"일 확률이 높음)
    #         order = np.argsort(vals)
    #         sorted_vals = vals[order]
    #         sorted_labels = labels[order]

    #         # 실제 False 수
    #         n_false = np.sum(sorted_labels == 1)

    #         # 뒤에서 n_false개가 모두 False면 perfect → 실제로 몇 개 맞췄는지 계산
    #         predicted_false_mask = np.zeros_like(sorted_labels, dtype=bool)
    #         predicted_false_mask[-n_false:] = True

    #         acc = np.mean(sorted_labels[predicted_false_mask] == 1)
    #         acc_per_class[class_name] = acc
    #         from sklearn.metrics import roc_auc_score
    #         auc = roc_auc_score(labels, vals)
    #         auc_per_class[class_name] = auc

    #         print(f"{class_name:<25} | False={len(sub_false):<4} Hard={len(sub_hard):<4} | Acc={acc*100:5.2f}%")

    #     # 평균 accuracy
    #     if len(acc_per_class) > 0 and len(auc_per_class) > 0:
    #         mean_acc = np.mean(list(acc_per_class.values()))
    #         print(f"\n✅ Mean Hard-vs-False Accuracy: {mean_acc*100:.2f}%")

    #         if logger is not None:
    #             logger.experiment.log({
    #                 f"{step_tag}/hard_vs_false_acc_mean": mean_acc,
    #                 **{f"AUC/hard_vs_false_acc_{cls}": auc for cls, auc in auc_per_class.items()},
    #                 **{f"{step_tag}/hard_vs_false_acc_{cls}": acc for cls, acc in acc_per_class.items()}
    #             })


    #     try:
    #         print("\n[False vs Hard Rank Distribution Plot]\n")

    #         false_wvals = W_final_np[false_mask]
    #         hard_wvals  = W_final_np[hard_mask]

    #         if len(false_wvals) > 0 and len(hard_wvals) > 0:
    #             # False=0, Hard=1 로 명시적으로 지정
    #             vals = np.concatenate([false_wvals, hard_wvals])
    #             labels = np.concatenate([np.zeros_like(false_wvals), np.ones_like(hard_wvals)])  
    #             order = np.argsort(vals)

    #             sorted_labels = labels[order]
    #             sorted_vals = vals[order]

    #             plt.figure(figsize=(10, 2.5))
    #             plt.scatter(
    #                 np.arange(len(sorted_vals)),
    #                 np.zeros_like(sorted_vals),
    #                 c=["royalblue" if l == 0 else "red" for l in sorted_labels],
    #                 s=8,
    #                 alpha=0.8,
    #                 edgecolors="none",
    #             )

    #             plt.title("False (blue) vs Hard (red) Order by $W_{final}$", fontsize=13)
    #             plt.xlabel("Rank (sorted by $W_{final}$)")
    #             plt.yticks([])
    #             plt.tight_layout()

    #             if logger:
    #                 logger.experiment.log({
    #                     f"{step_tag}/false_vs_hard_rank_order": wandb.Image(plt.gcf())
    #                 })
    #             plt.close()
    #             print(f"✅ False vs Hard rank order plot logged ({step_tag})")
    #         else:
    #             print("⚠️ Insufficient False/Hard samples for rank visualization.")

    #     except Exception as e:
    #         print("ordering error", e)


    #     # ===============================================================
    #     # 3️⃣ 통계 헬퍼
    #     # ===============================================================
    #     def compute_stats(vals):
    #         if vals.size == 0:
    #             return {k: np.nan for k in ["mean", "median", "min", "max", "q1", "q3", "count"]}
    #         return {
    #             "mean": np.mean(vals),
    #             "median": np.median(vals),
    #             "min": np.min(vals),
    #             "max": np.max(vals),
    #             "q1": np.quantile(vals, 0.25),
    #             "q3": np.quantile(vals, 0.75),
    #             "count": vals.size,
    #         }

    #     def compute_all_stats(mat, mask):
    #         out = {
    #             "W_app": compute_stats(W_app_np[mask]),
    #             "W_damp": compute_stats(W_damp_np[mask]),
    #             "W_final": compute_stats(W_final_np[mask]),
    #             "Stable": compute_stats(sim_stb_np[mask]),
    #             "Vid_mom": compute_stats(sim_vid_np[mask]),
    #             "Sen_mom": compute_stats(sim_sen_np[mask]),
    #             "Cross_Sim": compute_stats(sim_cross_np[mask]),
    #             "MotDamp": compute_stats(motion_damp_np[mask]),
    #         }
    #         if sim_vid_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
    #             out["Vid_mom_raw"] = compute_stats(sim_vid_raw_np[mask])
    #         if sim_sen_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
    #             out["Sen_mom_raw"] = compute_stats(sim_sen_raw_np[mask])
    #         if sim_z_vid_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
    #             out["Z_vid_mom_raw"] = compute_stats(sim_z_vid_raw_np[mask])
    #         if sim_z_sen_raw_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
    #             out["Z_sen_mom_raw"] = compute_stats(sim_z_sen_raw_np[mask])
    #         if sim_app_vid_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
    #             out["Sim_app_vid"] = compute_stats(sim_app_vid_np[mask])
    #         if sim_app_sen_np is not None:  # ✅ (2) 정규화되지 않은 sensor similarity 추가
    #             out["Sim_app_sen"] = compute_stats(sim_app_sen_np[mask])
    #         return out

    #     # ===============================================================
    #     # 4️⃣ Class-wise 통계 계산
    #     # ===============================================================
    #     print(f"\n[{step_tag}] Composite-class (spatial×motion) region stats\n")
    #     print(f"{'Class':<20} | {'Type':<6} | {'Cnt':<6} | {'W_app':<6} | {'W_dmp':<6} | {'W_fin':<6} | {'Stbl':<6} | {'Vid':<6} | {'Sen':<6} | {'Sen_mom_raw':<6} | {'Z_sen_mom_raw':<6} | {'Cros':<6} | {'Sim_app_sen':<10} | {'Sim_app_vid':<10} ")
    #     #print(f"{'Class':<20} | {'Type':<6} | {'Cnt':<6} | {'W_app':<6} | {'W_dmp':<6} | {'W_fin':<6} | {'Stbl':<6} | {'Vid':<6} | {'Sen':<6} | {'Vid_mom_raw':<6} | {'Sen_mom_raw':<6} | {'Z_vid_mom_raw':<6} | {'Z_sen_mom_raw':<6} | {'Cros':<6}")
    #     print("-" * 195)

    #     wandb_data = {}

    #     for class_idx, class_name in enumerate(class_names):
    #         # 현재 클래스 샘플 인덱스
    #         class_mask = (labels_np == class_idx)
    #         if not np.any(class_mask):
    #             continue

    #         # 이 클래스의 샘플들이 포함된 페어
    #         class_pair_mask = (class_mask[:, None] | class_mask[None, :]) & mask_upper

    #         # Hard/False 교집합
    #         hard_pairs = hard_mask & class_pair_mask
    #         false_pairs = false_mask & class_pair_mask
    #         # print("mean and median")
    #         # print("q1 and q3")

    #         region_dict = {
    #             "Hard": hard_mask & class_pair_mask,
    #             "False": false_mask & class_pair_mask,
    #             "Easy(o)": motion_same_mask & class_pair_mask,
    #             "Easy(x)": motion_diff_mask & class_pair_mask,
    #         }

    #         for region_name, region_mask in region_dict.items():
    #             stats = compute_all_stats(W_app_np, region_mask)
    #             count = stats["W_app"]["count"]
    #             if count == 0:
    #                 continue

    #             print(f"{class_name:<20} | {region_name:<6} | {count:<6} | "
    #               f"{stats['W_app']['mean']:<6.2f} | "
    #               f"{stats['W_damp']['mean']:<6.2f} | "
    #               f"{stats['W_final']['mean']:<6.2f} | "
    #               f"{stats['Stable']['mean']:<6.2f} | "
    #               f"{stats['Vid_mom']['mean']:<6.2f} | "
    #               f"{stats['Sen_mom']['mean']:<6.2f} | "
    #             #   f"{(stats['Vid_mom_raw']['mean'] if 'Vid_mom_raw' in stats else np.nan):<15.2f} | "
    #               f"{(stats['Sen_mom_raw']['mean'] if 'Sen_mom_raw' in stats else np.nan):<15.2f} | "
    #             #   f"{(stats['Z_vid_mom_raw']['mean'] if 'Z_vid_mom_raw' in stats else np.nan):<15.2f} | "
    #               f"{(stats['Z_sen_mom_raw']['mean'] if 'Z_sen_mom_raw' in stats else np.nan):<15.2f} | "
    #               f"{stats['Cross_Sim']['mean']:<6.2f} | "
    #               f"{(stats['Sim_app_vid']['mean'] if 'Sim_app_vid' in stats else np.nan):<10.2f} | "
    #               f"{(stats['Sim_app_sen']['mean'] if 'Sim_app_sen' in stats else np.nan):<10.2f}")

    #         print("--------------------------")
        

        print(f"\n✅ [{step_tag}] composite class-based symmetric region stats computed.")

        visualize_Wfinal_differences(
            W_final_np=W_final_np,
            sim_stb_np=sim_stb_np,
            hard_mask=hard_mask,
            false_mask=false_mask,
            motion_same_mask=motion_same_mask,
            labels_np=labels_np,
            class_names=class_names,
            step_tag=step_tag,
            epoch=self.epoch,
            logger=logger
        )