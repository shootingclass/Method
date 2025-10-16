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
from baseline_modules.base import BasePretrainModule

####################################################################



class MethodLightningModule(BasePretrainModule):

    def __init__(self, args, train_dataloader):
        super().__init__(args)
        # 1. 모델 구성 요소 초기화

        self.video_model = VisionModel(image_size=224)
        self.sensor_model = SensorModel(sensor_channels=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.clustering_model = ClusteringModel(
            encoder=self.sensor_model,
            embedding_dim=self.hparams.embedding_dim,
            num_sensors=self.hparams.num_sensors,
            num_clusters=self.hparams.num_classes,
            train_dataloader=train_dataloader,
            top_k=self.hparams.top_k,
            prototype_cache_dir=os.path.join(self.hparams.cache_dir, "prototypes"),
            dataset_name=self.hparams.dataset_name,
        )
        self.train_dataloader = train_dataloader
        self.epoch = 0
        
        self.appearance_classifier = nn.Linear(256, self.hparams.num_classes) 
    
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.success_labels=[0 for i in range(self.hparams.num_classes)]
        self.fail_labels=[0 for i in range(self.hparams.num_classes)]
        print(self.global_rank, "Model initialized.")

    
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

        # 실제 배치 크기를 텐서에서 직접 가져옵니다.
        current_batch_size = videos.size(0)

        self.clustering_model.train()
        self.video_model.train()
        self.sensor_model.train()

        # --- 1. 클러스터링 단계 ---
        
        # 데이터 증강 적용 (선택적)
        if self.clustering_model.epoch > 1:  # 첫 에폭은 원본 데이터로 클러스터링
            imu_data_aug = self.clustering_model.augment_imu_data(sensors)
        else:
            imu_data_aug = sensors
        
        # 모델 순전파
        scores, features = self.clustering_model(imu_data_aug, return_features=True, labels=labels, sample_ids=sample_ids)
        
        # 메모리 뱅크에서 저장된 pseudo label 가져오기
        stored_pseudo_labels = self.clustering_model.get_pseudo_labels(sample_ids)
        
        # 클러스터 할당 (argmax) - ODC 논문과 유사하게 메모리 뱅크 활용
        with torch.no_grad():
            # 1. 현재 모델의 예측값 계산
            current_cluster_ids = torch.argmax(scores, dim=1)

            # 2. EMA 방식으로 메모리 뱅크 업데이트 (레이블과 특징 모두)
            #    (이 함수는 내부적으로 old_feature와 new_feature를 섞어줍니다)
            self.clustering_model.clustering_manager.update_samples_memory(sample_ids, 
                                                    features)

            # 손실 계산에 사용할 레이블은 안정적인 '과거'의 레이블을 사용
            # get_pseudo_labels는 업데이트 '전'의 레이블을 반환해야 함
            loss_labels = stored_pseudo_labels.to(self.device)
            # 만약 첫 방문 샘플(-1)이 있다면, 현재 예측값을 임시로 사용
            mask_new = (loss_labels == -1)
            if mask_new.any():
                loss_labels[mask_new] = current_cluster_ids[mask_new]
        # # 클래스 가중치 계산
        class_weights = self.clustering_model.clustering_manager.compute_class_weights()
        
        # # ODC 손실 계산 (cross-entropy) - 저장된 pseudo label과 클래스 가중치 사용
        scores = F.normalize(scores, dim=1)
        loss_cluster = F.cross_entropy(scores, loss_labels, weight=class_weights.to(self.device))
        


        # print("centroids updated", self.clustering_model.clustering_manager.centroids)
        pseudo_labels_all = self.all_gather(loss_labels).view(-1)

        # 통계 수집
        # # --- 2. 에포크(Epoch) 기반의 조건부 로직 ---
        # # self.epoch을 사용하여 현재 에포크를 확인합니다.
        if self.epoch < self.hparams.threshold_epoch:
            output = {
                'features': features.detach(),
                'labels': labels.detach(),
                'predicted_labels': loss_labels.detach()
            }
            self.training_steps_outputs.append(output)
            return loss_cluster
            # # 최종 손실을 반환하면 Lightning이 알아서 backward 및 step을 수행합니다.
            
            # cropped_videos, iou_scores = self.video_model.patch_selection(videos, labels)
            # # 모델의 forward pass를 크롭된 비디오로 수행
            # # 이제부터는 'cropped_videos'를 사용합니다.
            # logits = self.video_model(cropped_videos)['logits']
            
            # # 'iou_scores'를 사용해 yolo_loss를 계산합니다.
            # yolo_loss = 1.0 - iou_scores.mean()
            
            # self.log('avg_iou', iou_scores.mean())

            # # 3. 최종 Loss 계산 및 학습
            # accuracy = (logits.argmax(dim=1) == labels).float().mean()
            # loss_patch = F.cross_entropy(logits, labels)
            # loss = loss_patch + yolo_loss
            # # loss = loss_patch + loss_cluster
            # fail_label = (logits.argmax(dim=1) != labels)
            # success_label = (logits.argmax(dim=1) == labels)
            # successed_true_labels = labels[success_label].tolist()
            # for success_label in successed_true_labels:
            #     self.success_labels[success_label] += 1
            # # boolean Tensor를 이용해 틀린 예측에 해당하는 실제 정답 레이블을 추출
            # failed_true_labels = labels[fail_label].tolist()
            # for fail_label in failed_true_labels:
            #     self.fail_labels[fail_label] += 1

            # if self.global_rank == 0:
            #     metrics_to_log = {
            #         'train_acc': accuracy,
            #         'train_loss': loss,
            #     }
                
            #     # 틀린 레이블이 있을 경우에만 히스토그램을 로그
            #     if failed_true_labels:
            #         print(f"Logging failed labels histogram with {len(failed_true_labels)} entries.")
            #         # 'failed_labels_dist'라는 이름으로 히스토그램을 생성하여 기록
            #         print("success labels:", self.success_labels)
            #         print("failed labels:", self.fail_labels)
            #         metrics_to_log['failed_labels_dist'] = wandb.Histogram(failed_true_labels)
                    
            #     wandb.log(metrics_to_log)
            
            return loss

        # --- 3. 분리(Disentanglement) 단계 ---
        model_output = self.video_model(videos)
        v_motion = model_output['v_motion']
        v_appearance = model_output['v_appearance']

        sensor_output = self.sensor_model(sensors)
        sensor_emb = sensor_output['emb']

        # 모든 GPU의 임베딩을 self.all_gather로 수집하고 합칩니다.
        v_motion_all = self.all_gather(v_motion).view(-1, 256)
        sensor_emb_all = self.all_gather(sensor_emb).view(-1, 256)
        v_appearance_all = self.all_gather(v_appearance).view(-1, v_appearance.shape[-1]) # (B, 256)

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

        # Appearance-based Classification Loss
        appearance_logits = self.appearance_classifier(v_appearance_all)
        loss_appearance_clf = F.cross_entropy(appearance_logits, pseudo_labels_all)

        # 최종 손실 계산
        lambda_cluster = 0.1
        lambda_ortho = 0.1
        lambda_info_nce = 0.1
        lambda_triplet = 0.1
        lambda_appearance_clf = 0.6

        final_loss = (lambda_cluster * loss_cluster +
                    lambda_info_nce * info_nce_loss +
                    lambda_ortho * ortho_loss +
                    lambda_triplet * triplet_loss +
                    lambda_appearance_clf * loss_appearance_clf)

        # final_loss = (lambda_cluster * loss_cluster +
        #             lambda_info_nce * info_nce_loss +
        #             lambda_ortho * ortho_loss +
        #             lambda_triplet * triplet_loss)



        # 개별 및 최종 손실을 로깅합니다.
        self.log('train/loss_cluster', loss_cluster, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/info_nce_loss', info_nce_loss, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/ortho_loss', ortho_loss, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/triplet_loss', triplet_loss, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/loss_appearance_clf', loss_appearance_clf, batch_size=current_batch_size, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train/final_loss', final_loss, batch_size=current_batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
        
        # self.global_rank가 0일 때 (즉, 마스터 프로세스일 때)만 로깅 코드를 실행합니다.
        # if self.global_rank != 0 :
        #     self.eval()
        #     self.clustering_model.evaluate(self.device)
            # print("go eval with ", self.global_rank)
        print("evaluate called on rank", self.global_rank)
        # 매 2 에폭마다 훈련 데이터셋에 대한 클러스터링 성능 평가
        
        if self.global_rank == 0:
            if self.epoch >= self.hparams.threshold_epoch:  
                with torch.no_grad():
                    batch = next(iter(self.train_dataloader()))

                    videos, _, _, _ = batch
                    videos = videos.to(self.device)

                    # 시각화는 첫 번째 비디오만 사용
                    video_to_log = videos[0]

                    model_output = self.video_model(videos)

                    # 원본 비디오 준비 
                    mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
                    
                    # wandb.Video 형식 (T, C, H, W)에 맞게 차원 변경
                    original_video_frames = video_to_log

                    # 역정규화 수행 (std 곱하고 mean 더하기)
                    unnormalized_video = original_video_frames * std + mean
                    
                    # 값 범위를 0-1 사이로 안전하게 클리핑
                    unnormalized_video = torch.clamp(unnormalized_video, 0, 1)
                    video_original = (unnormalized_video * 255).to(torch.uint8)

                    # final_features (전체) 시각화 
                    final_features = model_output['final_features'][0]
                    
                    # final_features의 shape는 (T, Num_Tokens, Feature_Dim) 형태입니다.
                    # Num_Tokens 차원(dim=1)에서 첫 번째 토큰(CLS)을 제외하고 나머지를 선택합니다.
                    features_no_cls = final_features[:, 1:, :]
                    
                    # 이제 CLS 토큰이 제외된 피처맵을 사용해 패치 수를 계산합니다.
                    num_patches = features_no_cls.shape[1]
                    grid_size = int(math.sqrt(num_patches))
                    
                    if grid_size * grid_size == num_patches:
                        final_features_grid = features_no_cls.reshape(features_no_cls.shape[0], grid_size, grid_size, -1)
                        final_features_permuted = final_features_grid.permute(0, 3, 1, 2)
                        heatmap_final = torch.mean(final_features_permuted, dim=1)

                        # 히트맵을 원본 비디오 크기로 업샘플링(확대)합니다.
                        H, W = video_original.shape[2], video_original.shape[3] # 원본 비디오의 높이(H)와 너비(W)
                        
                        # interpolate를 위해 채널 차원(C)을 임시로 추가 (T, 1, 7, 7)
                        heatmap_unsqueezed = heatmap_final.unsqueeze(1) 
                        
                        # 'bilinear' 방식으로 부드럽게 크기를 확대합니다.
                        heatmap_upsampled = F.interpolate(heatmap_unsqueezed, size=(H, W), mode='bilinear', align_corners=False)
                        
                        # 임시로 추가했던 채널 차원을 다시 제거 (T, H, W)
                        heatmap_upsampled = heatmap_upsampled.squeeze(1)

                        # 업샘플링된 히트맵에 컬러맵을 적용합니다.
                        normalized_frames = []
                        for frame in heatmap_upsampled: # 이제 고해상도 히트맵을 사용합니다.
                            min_val, max_val = torch.min(frame), torch.max(frame)
                            normalized_frame = (frame - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(frame)
                            normalized_frames.append(normalized_frame)
                        heatmap_colored = torch.stack(normalized_frames)

                    cmap = plt.get_cmap('viridis')

                    # (T, H, W, 3) 형태로 변환, 값은 0~1 사이
                    colored_frames_np = cmap(heatmap_colored.cpu().numpy())[:, :, :, :3] 

                    # (T, 3, H, W) 형태로 변환, 값은 0~255 사이
                    colored_video_tensor = (torch.from_numpy(colored_frames_np).permute(0, 3, 1, 2) * 255).to(torch.uint8)

                    # 원본 비디오와 컬러 히트맵을 오버레이(겹치기)합니다.
                    # 연산을 위해 원본 비디오와 히트맵을 float 형태로 변경
                    original_float = video_original.cpu().to(torch.float32)
                    heatmap_float = colored_video_tensor.to(torch.float32)

                    # 가중치를 주어 두 비디오를 합칩니다.
                    alpha = 0.5
                    overlayed_video = original_float * (1 - alpha) + heatmap_float * alpha

                    # wandb 로깅을 위해 다시 uint8 형태로 변환
                    video_final_features_overlay = overlayed_video.to(torch.uint8)

                    # transformed_features (부분) 시각화
                    transformed_features = model_output['transformed_features'][0]
                    transformed_features_permuted = transformed_features.permute(1, 0, 2, 3)
                    heatmap_transformed = torch.mean(transformed_features_permuted, dim=1)

                    normalized_frames_transformed = []
                    for frame in heatmap_transformed:
                        min_val, max_val = torch.min(frame), torch.max(frame)
                        normalized_frame = (frame - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(frame)
                        normalized_frames_transformed.append(normalized_frame)
                    heatmap_transformed = torch.stack(normalized_frames_transformed)

                    cmap = plt.get_cmap('viridis')
                    colored_frames_transformed_np = cmap(heatmap_transformed.cpu().numpy())[:, :, :, :3]
                    colored_video_transformed_tensor = torch.from_numpy(colored_frames_transformed_np).permute(0, 3, 1, 2)
                    video_transformed_features = (colored_video_transformed_tensor * 255).to(torch.uint8)

                    # Wandb 로깅
                    log_dict = {
                        "epoch_visuals/transformed_features": wandb.Video(video_transformed_features.cpu(), fps=4, format="gif")
                    }
                    if video_final_features_overlay is not None:
                        log_dict["epoch_visuals/final_features"] = wandb.Video(video_final_features_overlay.cpu(), fps=4, format="gif")

                    self.logger.experiment.log(log_dict)
                    print("Feature map visualizations logged to wandb.")
            
            print("threshold_epoch not reached, skipping feature visualization.")
        
            self.train()  # 모델을 다시 훈련 모드로 설정
        print("Epoch end processing completed on rank", self.global_rank)