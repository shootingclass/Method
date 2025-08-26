import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from yaml import warnings # tqdm 라이브러리 임포트 추가
import wandb
import os
import torch.distributed as dist
import warnings

# model.py에 저장된 모델 클래스를 임포트합니다.
from visualization import save_video_grid


####################################################################


def calculate_sensor_stats(dataset):
    """
    Calculates mean and std for each sensor channel across the entire dataset.
    """
    all_sensor_data = []
    print("Calculating sensor statistics...")
    
    for i in tqdm(range(len(dataset)), desc="Collecting sensor data"):
        _, sensor_data, _ = dataset[i]        
        all_sensor_data.append(sensor_data)
    
    concatenated_data = np.concatenate(all_sensor_data, axis=1)
    
    mean = np.mean(concatenated_data, axis=1)
    std = np.std(concatenated_data, axis=1)
    
    print("Calculation complete.")
    return {'mean': mean, 'std': std}


#################################################################


def save_stats(stats, path):
    """
    Saves the calculated statistics to a .npy file.
    """
    np.save(path, stats)
    print(f"Sensor statistics saved to {path}")


#################################################################


def load_stats(path):
    """
    Loads statistics from a .npy file.
    """
    stats = np.load(path, allow_pickle=True).item()
    print(f"Sensor statistics loaded from {path}")
    return stats


#################################################################


def train_one_epoch(video_model, sensor_model, clustering_model, dataloader, optimizer, device, epoch, output_dir, rank):
    video_model.train()
    sensor_model.train()

    total_loss = 0.0
    temperature = 0.07 # InfoNCE Loss의 temperature

    if rank == 0:
        iterable = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    else:
        iterable = dataloader

    for batch_idx, (videos, sensors, labels, _) in enumerate(iterable):
        
        videos = videos.to(device)
        sensors = sensors.to(device)

        optimizer.zero_grad()
        
        
        ####### Clustering #######
        num_sensors = sensors.shape[1]
        rule_based_feature = clustering_model.module.get_representative_sensor_feature(sensors, labels, num_sensors)
        scores_cluster, final_feature_cluster, alpha = clustering_model(sensors, rule_based_feature)
        scores_sk = clustering_model.module.sinkhorn_knopp(scores_cluster)
    
        with torch.no_grad():
            pseudo_labels = torch.argmax(scores_sk, dim=1)

        # 클러스터링 피처와 레이블을 모든 GPU에서 수집합니다.
        world_size = dist.get_world_size()
        local_batch_size = final_feature_cluster.shape[0]
        global_batch_size = local_batch_size * world_size

        # 결과를 담을 큰 텐서를 미리 할당합니다.
        final_feature_cluster_all = torch.empty(
            global_batch_size, 
            final_feature_cluster.shape[1], 
            device=device, 
            dtype=final_feature_cluster.dtype
        )
        pseudo_labels_all = torch.empty(global_batch_size, device=device, dtype=pseudo_labels.dtype)

        # all_gather_into_tensor 함수로 한 번에 모읍니다.
        dist.all_gather_into_tensor(final_feature_cluster_all, final_feature_cluster)
        dist.all_gather_into_tensor(pseudo_labels_all, pseudo_labels)

        # 글로벌 배치 기준으로 MSE Loss를 다시 계산합니다.
        # 모든 GPU의 피처들이 자신이 속한 클러스터의 중심(prototypes)에 가까워지도록 합니다.
        mse_loss = F.mse_loss(final_feature_cluster_all, clustering_model.module.prototypes[pseudo_labels_all])
        # -----------------------------------------------------------

        prototypes = clustering_model.module.prototypes
        p1 = prototypes.unsqueeze(1)
        p2 = prototypes.unsqueeze(0)
        mse_matrix = F.mse_loss(p1, p2, reduction='none').mean(dim=2)
        n_proto = clustering_model.module.num_clusters     

        # 클러스터 중심들이 서로 멀어지도록 만드는 손실
        # 이 Loss의 목적은 프로토타입 벡터들이 서로 멀리 떨어지게 만드는 것이므로, 입력 데이터 배치와는 독립적
        # 각 GPU는 동일한 모델 복사본을 가지고 시작하며, backward() 호출 시 그래디언트 동기화를 통해 모든 모델이 동일하게 업데이트됩니다. prototypes가 모델의 파라미터(nn.Parameter)라면 이 원칙이 그대로 적용됨
        # 동일한 입력값(동일한 프로토타입들)으로 동일한 계산을 수행하므로, 그 결과는 모든 GPU에서 완전히 동일
        # Loss 계산에 데이터 배치가 관여하지 않으므로, 데이터를 모으는 것은 불필요한 연산임
        diversity_loss = - (mse_matrix.sum()) / (n_proto * (n_proto - 1))
        
        loss_cluster = mse_loss + diversity_loss

        # 9-epoch까지는 클러스터링 모델만 학습
        if epoch < clustering_model.module.threshold_epoch: 
            loss_cluster.backward()
            optimizer.step()
            total_loss += loss_cluster.item()
            continue
    
        
        ####### Disentangle #######
        model_output = video_model(videos)
        v_motion = model_output['v_motion']
        v_appearance = model_output['v_appearance']

        sensor_output = sensor_model(sensors)
        sensor_emb = sensor_output['emb']

        # 모든 GPU의 임베딩을 하나로 모읍니다.
        world_size = dist.get_world_size()

        # 각 GPU의 배치 사이즈를 가져옵니다.
        batch_size = v_motion.shape[0]
        global_batch_size = batch_size * world_size

        # 결과를 담을 큰 텐서를 미리 할당합니다.
        v_motion_all = torch.empty(global_batch_size, v_motion.shape[1], device=device, dtype=v_motion.dtype)
        sensor_emb_all = torch.empty(global_batch_size, sensor_emb.shape[1], device=device, dtype=sensor_emb.dtype)
        v_appearance_all = torch.empty(global_batch_size, v_appearance.shape[1], device=device, dtype=v_appearance.dtype)

        # all_gather_into_tensor 함수로 한 번에 모읍니다.
        dist.all_gather_into_tensor(v_motion_all, v_motion)
        dist.all_gather_into_tensor(sensor_emb_all, sensor_emb)
        dist.all_gather_into_tensor(v_appearance_all, v_appearance)
        # ---------------------------------------------

        # --- InfoNCE Loss 계산 (수정된 부분) ---
        # Global Batch 임베딩으로 계산합니다.
        v_motion_norm = F.normalize(v_motion_all, p=2, dim=1)
        sensor_emb_norm = F.normalize(sensor_emb_all, p=2, dim=1)
        sim_matrix = torch.matmul(v_motion_norm, sensor_emb_norm.T) / temperature
        labels = torch.arange(sim_matrix.size(0), device=device) # 레이블은 Global Batch 크기에 맞게 생성합니다.
        info_nce_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2

        # --- 직교성 제약 Loss 계산 (수정된 부분) ---
        v_appearance_norm = F.normalize(v_appearance_all, p=2, dim=1)
        cosine_similarity = (v_motion_norm * v_appearance_norm).sum(dim=1)
        ortho_loss = (cosine_similarity ** 2).mean()

        # --- Triplet Loss with Hard Negative Mining ---
        # 1. 모든 쌍의 거리 계산 (Distance Matrix 생성)
        # v_motion_all과 prototypes 사이의 유클리드 거리를 계산합니다.
        # dist_matrix 크기: [global_batch_size, num_clusters]
        dist_matrix = torch.cdist(v_motion_all, clustering_model.prototypes, p=2)

        # 2. Positive 샘플 거리 추출 및 마스킹 준비
        # 각 v_motion에 해당하는 Positive 프로토타입과의 거리를 가져옵니다.
        # positive_distances 크기: [global_batch_size]
        positive_distances = dist_matrix.gather(1, pseudo_labels_all.unsqueeze(1)).squeeze()

        # 3. Hard Negative 선택 (최소값 찾기)
        # Positive 위치에 아주 큰 값을 더해 검색에서 제외합니다.
        masked_dist_matrix = dist_matrix.clone()
        masked_dist_matrix.scatter_(1, pseudo_labels_all.unsqueeze(1), float('inf'))

        # 마스킹된 거리 행렬에서 각 v_motion에 가장 가까운 Negative 프로토타입의 거리를 찾습니다.
        # hard_negative_distances 크기: [global_batch_size]
        hard_negative_distances = torch.min(masked_dist_matrix, dim=1).values

        # 4. Triplet Loss 계산
        # L(a, p, n) = max(d(a, p) - d(a, n) + margin, 0)
        margin = 1.0
        triplet_loss = torch.relu(positive_distances - hard_negative_distances + margin).mean()

        # --- 최종 손실 계산 (로직 동일) ---
        lambda_cluster = 0.5
        lambda_ortho = 0.1
        lambda_info_nce = 0.9
        lambda_triplet = 0.1
        final_loss = lambda_cluster * loss_cluster + lambda_info_nce * info_nce_loss + lambda_ortho * ortho_loss + lambda_triplet * triplet_loss

        # --- 역전파 및 파라미터 업데이트 ---
        # DDP가 모든 GPU에 걸쳐 그래디언트를 자동으로 동기화하고 평균냅니다.
        final_loss.backward()
        optimizer.step()

        total_loss += final_loss.item()

    # 평균 손실 계산
    avg_loss = total_loss / len(dataloader)

    # ------------------ 수정 제안 ------------------
    # 1. 현재 프로세스의 손실 값을 텐서로 만듭니다.
    loss_tensor = torch.tensor([avg_loss], dtype=torch.float32, device=device)

    # 2. all_reduce를 통해 모든 프로세스의 loss_tensor 값을 더합니다.
    # op=dist.ReduceOp.SUM은 모든 텐서의 값을 합산하는 연산입니다.
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

    # 3. 전체 GPU의 수(world_size)로 나누어 평균을 계산합니다.
    # all_reduce 후에는 loss_tensor[0]에 모든 GPU의 손실 합이 저장됩니다.
    avg_loss_all_gpus = loss_tensor.item() / dist.get_world_size()

    return avg_loss_all_gpus




