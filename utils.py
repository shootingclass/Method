import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm # tqdm 라이브러리 임포트 추가
import wandb
import os

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

    # 하이퍼파라미터: InfoNCE Loss의 temperature
    temperature = 0.07

    # rank 0 프로세스에서만 tqdm 프로그레스 바를 사용합니다.
    if rank == 0:
        iterable = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    else:
        iterable = dataloader

    for batch_idx, (videos, sensors, labels, _) in enumerate(iterable):
        
        videos = videos.to(device)
        sensors = sensors.to(device)

        optimizer.zero_grad()
        
        # 0. 클러스터링 모델 순전파 및 손실 계산 (비지도 학습)
        num_sensors = sensors.shape[1]
        rule_based_feature = clustering_model.get_representative_sensor_feature(sensors, labels, num_sensors)
        scores_cluster, final_feature_cluster, alpha = clustering_model(sensors, rule_based_feature)
        scores_sk = clustering_model.sinkhorn_knopp(scores_cluster)
        # wandb.log({"train_alpha": alpha})
    
        with torch.no_grad():
            pseudo_labels = torch.argmax(scores_sk, dim=1)
            
        mse_loss = F.mse_loss(final_feature_cluster, clustering_model.prototypes[pseudo_labels])
        
        prototypes = clustering_model.prototypes
        p1 = prototypes.unsqueeze(1)
        p2 = prototypes.unsqueeze(0)
        mse_matrix = F.mse_loss(p1, p2, reduction='none').mean(dim=2)
        n_proto = clustering_model.num_clusters
        diversity_loss = - (mse_matrix.sum()) / (n_proto * (n_proto - 1))
        
        loss_cluster = mse_loss + diversity_loss

        if epoch < clustering_model.threshold_epoch:
            # 9-epoch까지는 클러스터링 모델만 학습
            loss_cluster.backward()
            optimizer.step()
            total_loss += loss_cluster.item()
            continue
        # 일단 rule-based feature 사용 (almost 0.7 accuracy)
        labels = pseudo_labels
        prototypes = clustering_model.prototypes

        # 1. 각 모델에서 임베딩 추출
        # DDP로 래핑된 모델은 내부적으로 .module을 호출하므로 직접적인 접근은 필요 없습니다.
        model_output = video_model(videos)
        v_motion = model_output['v_motion']
        v_appearance = model_output['v_appearance']

        sensor_output = sensor_model(sensors)
        sensor_emb = sensor_output['emb']

        # --- InfoNCE Loss 계산 (로직 동일) ---
        v_motion_norm = F.normalize(v_motion, p=2, dim=1)
        sensor_emb_norm = F.normalize(sensor_emb, p=2, dim=1)
        sim_matrix = torch.matmul(v_motion_norm, sensor_emb_norm.T) / temperature
        labels = torch.arange(sim_matrix.size(0), device=device)
        info_nce_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2

        # --- 직교성 제약 Loss 계산 (로직 동일) ---
        v_appearance_norm = F.normalize(v_appearance, p=2, dim=1)
        cosine_similarity = (v_motion_norm * v_appearance_norm).sum(dim=1)
        ortho_loss = (cosine_similarity ** 2).mean()

        # --- 최종 손실 계산 (로직 동일) ---
        lambda_ortho = 0.1
        lambda_info_nce = 0.9
        final_loss = lambda_info_nce * info_nce_loss + lambda_ortho * ortho_loss

        # --- 역전파 및 파라미터 업데이트 ---
        # DDP가 모든 GPU에 걸쳐 그래디언트를 자동으로 동기화하고 평균냅니다.
        final_loss.backward()
        optimizer.step()

        total_loss += final_loss.item()

    # 각 프로세스별 평균 손실을 계산합니다.
    # 정확한 전체 평균을 원하면 all_reduce 연산이 필요하지만,
    # 일반적으로 rank 0의 값만으로도 충분히 경향을 파악할 수 있습니다.
    avg_loss = total_loss / len(dataloader)

    return avg_loss




