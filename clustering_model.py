import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import os
import yaml
import argparse
import wandb
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import MW2StackRNNPooling

# --- 사용자 정의 모듈 및 헬퍼 함수 (기존 코드와 동일) ---

START_INDEX = 1
END_INDEX = 11

def visualize_tsne_3D(embeddings, true_labels, pred_labels, prototypes=None, title="t-SNE Visualization", mapping=False):
    """
    t-SNE 결과를 3D로 시각화하고 Matplotlib Figure 객체를 반환합니다.
    """
    n_samples = prototypes.shape[0]
    label_names = [f"Class_{i}" for i in range(n_samples)]
    
    if n_samples <= 1:
        print(f"Warning: Cannot run t-SNE with {n_samples} samples.")
        return plt.figure()

    perplexity_value = min(30.0, float(n_samples - 1))
    if perplexity_value <= 0: perplexity_value = 1.0

    print(f"Running 3D t-SNE with {n_samples} samples and perplexity={perplexity_value:.1f}")
    
    # --- ✨ 핵심 수정: n_components=3 으로 변경 ---
    tsne = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, metric="cosine")
    
    if prototypes is not None:
        combined_data = np.vstack([embeddings, prototypes])
        reduced_data = tsne.fit_transform(combined_data)
        reduced_embeddings = reduced_data[:-len(prototypes)]
        reduced_prototypes = reduced_data[-len(prototypes):]
    else:
        reduced_embeddings = tsne.fit_transform(embeddings)
        reduced_prototypes = None

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(title, fontsize=16)

    # --- ✨ 핵심 수정: subplot을 3D로 설정 ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # 실제 레이블 기준 시각화
    scatter1 = ax1.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
        c=true_labels, cmap="tab10", alpha=0.7
    )
    ax1.set_title("True Labels")
    legend1_handles, _ = scatter1.legend_elements(num=n_samples)
    ax1.legend(legend1_handles, label_names)

    # 예측된 클러스터 기준 시각화
    scatter2 = ax2.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
        c=pred_labels, cmap="tab10", alpha=0.7
    )
    if mapping:
        ax2.set_title("Predicted Clusters (Mapped)")
    else:
        ax2.set_title("Predicted Clusters")
    
    if reduced_prototypes is not None:
        proto_labels = np.arange(len(prototypes))
        # 두 subplot에 모두 프로토타입을 표시
        ax1.scatter(
            reduced_prototypes[:, 0], reduced_prototypes[:, 1], reduced_prototypes[:, 2],
            c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5
        )
        ax2.scatter(
            reduced_prototypes[:, 0], reduced_prototypes[:, 1], reduced_prototypes[:, 2],
            c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5
        )
        
        # 프로토타입에 번호 추가
        for i in range(len(prototypes)):
            ax1.text(reduced_prototypes[i, 0], reduced_prototypes[i, 1], reduced_prototypes[i, 2], f'P{i}', fontsize=12, weight='bold')
            ax2.text(reduced_prototypes[i, 0], reduced_prototypes[i, 1], reduced_prototypes[i, 2], f'P{i}', fontsize=12, weight='bold')
    
    legend2_handles, _ = scatter2.legend_elements(num=n_samples)
    ax2.legend(legend2_handles, label_names)
        
    return fig
# LinearProbingEvaluator 클래스를 아래 코드로 교체하세요.
def visualize_tsne(embeddings, true_labels, pred_labels, title, prototypes=None, num_classes=7, mapping=False):
        """t-SNE 결과를 시각화하고 Matplotlib Figure 객체를 반환. 프로토타입도 함께 시각화 가능."""
        fig = visualize_tsne_3D(embeddings, true_labels, pred_labels, prototypes=prototypes, mapping=mapping)
        wandb.log({"t-SNE Visualization_3d": wandb.Image(fig, caption=title)})

        label_names = [f"Class_{i}" for i in range(num_classes)]
        assert num_classes == len(prototypes), f"num_classes must be equal to the number of prototypes, now num_classes: {num_classes}, len(prototypes): {len(prototypes)}"
        # --- ✨ 핵심 수정: t-SNE를 실행하기에 샘플 수가 충분한지 확인 ---
        if len(embeddings) <= 1:
            print(f"Warning: Cannot run t-SNE with {len(embeddings)} samples. Skipping visualization.")
            return plt.figure() # 빈 Figure 객체 반환

        # --- ✨ 프로토타입과 임베딩을 함께 변환하기 위해 결합 ---
        if prototypes is not None:
            combined_data = np.vstack([embeddings, prototypes])
        else:
            combined_data = embeddings

        # Perplexity는 샘플 수보다 작아야 함
        perplexity_value = min(30, len(combined_data) - 1)
        if perplexity_value <= 0: # 이중 안전장치
            perplexity_value = 1.0
        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, n_iter=300, metric="cosine")
        reduced_all = tsne.fit_transform(combined_data)
        
        reduced_embeddings = reduced_all[:len(embeddings)]
        if prototypes is not None:
            reduced_prototypes = reduced_all[len(embeddings):]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
        fig.suptitle(title, fontsize=16)
        
        # 실제 레이블 기준 시각화
        scatter1 = ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=true_labels, cmap="tab10", alpha=0.7)
        ax1.set_title("True Labels")

        # 예측된 클러스터 기준 시각화
        scatter2 = ax2.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=pred_labels, cmap="tab10", alpha=0.7)
        if mapping:
            ax2.set_title("Predicted Clusters (Mapped)")
        else:
            ax2.set_title("Predicted Clusters")
        
        # --- ✨ 프로토타입 시각화 추가 ---
        if prototypes is not None:
            proto_labels = np.arange(len(prototypes))
            # 두 subplot에 모두 프로토타입을 표시
            ax1.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5)
            ax2.scatter(reduced_prototypes[:, 0], reduced_prototypes[:, 1], c=proto_labels, cmap="tab10", marker='X', s=200, edgecolor='black', linewidth=1.5)
            
            # 프로토타입에 번호 추가
            for i in range(len(prototypes)):
                ax1.text(reduced_prototypes[i, 0] + 0.1, reduced_prototypes[i, 1] + 0.1, f'P{i}', fontsize=12, weight='bold')
                ax2.text(reduced_prototypes[i, 0] + 0.1, reduced_prototypes[i, 1] + 0.1, f'P{i}', fontsize=12, weight='bold')
            
            # 범례 업데이트
            handles1 = scatter1.legend_elements(num=num_classes)[0]
            proto_handle = plt.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='Prototypes')
            handles1.append(proto_handle)
            ax1.legend(handles=handles1, labels=label_names + ['Prototypes'])

            print("\n--- Legend Debugging Info ---")
            # 실제로 pred_labels에 어떤 값들이 들어있는지 확인
            unique_preds = np.unique(pred_labels)
            print(f"Unique predicted labels in data: {unique_preds}")
            print(f"Number of unique predicted labels: {len(unique_preds)}")

            # legend_elements가 생성하는 핸들의 실제 개수 확인
            handles_check = scatter2.legend_elements(num=num_classes)[0]
            print(f"Number of handles generated by legend_elements: {len(handles_check)}")
            print(f"Number of labels provided: {len(label_names) + 1}")
            print("---------------------------\n")
            handles2 = scatter2.legend_elements(num=num_classes)[0]
            handles2.append(proto_handle)
            print('handles2: ', handles2)
            print('label_names: ', label_names)
            ax2.legend(handles=handles2, labels=label_names + ['Prototypes'])
        else:
            assert False, "prototypes is not None"
            ax1.legend(handles=scatter1.legend_elements(num=num_classes)[0], labels=label_names)
            ax2.legend(handles=scatter2.legend_elements(num=num_classes)[0], labels=label_names)
        
        return fig
   

def get_sensor_name(sensor_index):
    return f"Sensor_{sensor_index}"

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_type="max", embedding_size=32):
        super().__init__()
        if pool_type == "max":
            pool_fn = torch.nn.MaxPool1d(kernel_size=3)
        elif pool_type == "adaptive":
            pool_fn = torch.nn.AdaptiveAvgPool1d(output_size=embedding_size)
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2,
                bias=False,
            ),
            pool_fn,
        )

    def forward(self, batch):
        return self.net(batch)

class HybridClusteringModule(nn.Module):
    def __init__(self, embedding_dim, num_sensors, num_clusters, prototypes=None, alpha_fixed=False, total_epochs=20, threshold_epoch=6):
        super().__init__()
        self.encoder = MW2StackRNNPooling(size_embeddings=embedding_dim, in_channels=num_sensors)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.9),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.projection_layer = nn.Linear(num_sensors, embedding_dim)
        if prototypes is not None:
            self.prototypes = nn.Parameter(prototypes)
        else:
            self.prototypes = nn.Parameter(torch.randn(num_clusters, embedding_dim))
        self.alpha_fixed = alpha_fixed
        self.current_epoch = 0
        self.threshold_epoch = threshold_epoch
        self.total_epochs = 20

    def forward(self, imu_batch, rule_based_feature, val=False):
        deep_embedding = self.encoder(imu_batch)['emb']
        if not self.alpha_fixed:
            raw_gate_score = self.gate(deep_embedding)
            alpha = raw_gate_score
            # 2. 온도(temperature)를 적용하여 Sigmoid 통과
            start_alpha = 0.5
            end_alpha = 1.0
            total_epochs = self.total_epochs

            # 현재 에폭 진행률
            progress = min(1.0, self.current_epoch / total_epochs) 
            # if progress > 0.7:
                # alpha_value=1.0
            # else:
                # alpha_value=0.5
            # alpha_value = raw_gate_score
            progress = 0.5 * (1 - torch.cos(torch.tensor(np.pi * self.current_epoch / total_epochs)))

            # alpha 값 계산
            alpha_value = start_alpha + (end_alpha - start_alpha) * progress
            alpha_value = alpha_value  + raw_gate_score *0.0
            if self.current_epoch > self.threshold_epoch:
                alpha_value = 1.0
            alpha = torch.tensor(alpha_value)
        else:
            alpha = torch.tensor(0.5, device=deep_embedding.device)

        if val:
            alpha = torch.tensor(1.0, device=deep_embedding.device)

        projected_rule_feature = self.projection_layer(rule_based_feature)
        final_feature = alpha * F.normalize(deep_embedding, dim=1) + (1 - alpha) * projected_rule_feature
        
        final_feature_norm = F.normalize(final_feature, dim=1)
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        scores = torch.matmul(final_feature_norm, prototypes_norm.t())
        return scores, final_feature, alpha

    def update_epoch(self, epoch):
        self.current_epoch = epoch

def get_representative_sensor_feature(imu_batch, labels, num_total_sensors=10, top_k=1):
    max_values, _ = torch.max(imu_batch, dim=2)
    min_values, _ = torch.min(imu_batch, dim=2)
    ranges = max_values - min_values

    min_range, _ = torch.min(ranges, dim=1, keepdim=True)
    max_range, _ = torch.max(ranges, dim=1, keepdim=True)
    weighted_features = (ranges - min_range) / (max_range - min_range + 1e-8)
    
    _, top_indices = torch.topk(weighted_features, k=top_k, dim=1)
    mask = torch.zeros_like(weighted_features)
    mask.scatter_(1, top_indices, 1)
    
    final_rule_feature = weighted_features * mask
    return final_rule_feature

def compute_hungarian_matching(pred_labels, true_labels, num_clusters):
    cost_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for i in range(len(pred_labels)):
        cost_matrix[pred_labels[i], true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = {i: j for i, j in zip(row_ind, col_ind)}
    return accuracy_score(true_labels, np.array([mapping.get(x, -1) for x in pred_labels])), mapping
# scikit-learn이 설치되어 있어야 합니다.
# pip install scikit-learn
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

# (기존 코드의 다른 함수들은 동일하다고 가정)

# def initialize_prototypes(dataloader, num_clusters, embedding_dim, num_sensors, device, args):
#     """라벨을 사용하지 않고 프로토타입 초기화"""
#     # 임시 인코더 및 프로젝션 레이어 생성
#     temp_encoder = MW2StackRNNPooling(size_embeddings=embedding_dim, in_channels=num_sensors).to(device)
#     temp_projection = nn.Linear(num_sensors, embedding_dim).to(device)
    
#     # 임베딩 수집
#     embeddings = []
    
#     print("Collecting embeddings for unsupervised prototype initialization...")
#     with torch.no_grad():
#         # 여러 배치에서 임베딩을 수집하여 데이터 분포를 더 잘 반영
#         for batch_idx, batch in enumerate(dataloader):
#             # 너무 많은 배치를 사용하지 않도록 제한 (예: 40개 배치)
#             if batch_idx >= 100: 
#                 break
                
#             imu_data = batch[1].to(device)
#             labels = batch[2].to(device)
            
#             # 규칙 기반 특징 및 딥러닝 임베딩 추출
#             rule_based_feature = get_representative_sensor_feature(imu_data, labels, num_total_sensors=num_sensors, top_k=args.top_k) # 라벨 사용 안함
#             deep_embedding = temp_encoder(imu_data)['emb']
#             rule_based_feature = temp_projection(rule_based_feature)
            
#             # 최종 임베딩 계산
#             final_feature = 0.5 * F.normalize(deep_embedding, dim=1) + 0.5 * rule_based_feature
#             embeddings.append(final_feature)
    
#     if not embeddings:
#         print("Warning: Dataloader is empty. Using random initialization.")
#         return torch.randn(num_clusters, embedding_dim, device=device)
        
#     embeddings = torch.cat(embeddings, dim=0).cpu().numpy() # KMeans는 CPU에서 동작
    
#     if len(embeddings) < num_clusters:
#         print(f"Warning: Only {len(embeddings)} embeddings collected, but {num_clusters} prototypes needed. Using random initialization.")
#         return torch.randn(num_clusters, embedding_dim, device=device)

#     print(f"Running K-Means++ to find initial {num_clusters} prototypes from {len(embeddings)} embeddings.")
    
#     # K-Means를 사용해 초기 프로토타입 추출 (init='k-means++'가 기본값)
#     # n_init='auto'는 최적의 초기화를 찾기 위해 여러 번 실행
#     kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=0)
#     kmeans.fit(embeddings)
    
#     # 추출된 프로토타입(군집 중심)을 텐서로 변환
#     prototypes = torch.from_numpy(kmeans.cluster_centers_).to(device)
    
#     # 정규화된 프로토타입 반환
#     return F.normalize(prototypes, dim=1)


def initialize_prototypes(dataloader, num_clusters, embedding_dim, num_sensors, device, args):
    """첫 번째 배치에서 프로토타입 초기화"""
    # 임시 인코더 생성
    temp_encoder = MW2StackRNNPooling(size_embeddings=embedding_dim, in_channels=num_sensors).to(device)
    temp_projection = nn.Linear(num_sensors, embedding_dim).to(device)
    
    # 임베딩 수집
    embeddings = []
    labels = []
    
    with torch.no_grad():
        # 첫 배치만 사용
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 40:  # 첫 배치만 사용
                break
                
            imu_data = batch[1].to(device)
            batch_labels = batch[2].to(device)
            label_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 8}
            batch_labels = torch.tensor([label_mapping[label.item()] for label in batch_labels])
            batch_labels = batch_labels.to(device)
            
            # 규칙 기반 특징 추출
            rule_based_feature = get_representative_sensor_feature(imu_data, batch_labels, num_total_sensors=num_sensors, top_k=args.top_k)
            
            # 딥러닝 임베딩 추출
            deep_embedding = temp_encoder(imu_data)['emb']
            
            # 규칙 기반 특징 투영
            rule_based_feature = temp_projection(rule_based_feature)
            
            # 최종 임베딩 계산 (alpha=0.5 사용)
            final_feature = 0.5 * F.normalize(deep_embedding, dim=1) + 0.5 * rule_based_feature
            
            embeddings.append(final_feature)
            labels.append(batch_labels)
    
    if not embeddings:  # 배치가 비어있을 경우 랜덤 초기화
        print("Warning: Empty batch, using random initialization")
        return torch.randn(num_clusters, embedding_dim)
        
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # 클래스별 평균 임베딩을 프로토타입으로 사용
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    # 초기 프로토타입 랜덤 초기화
    prototypes = torch.randn(num_clusters, embedding_dim)
    
    # 클래스 수가 프로토타입 수와 다를 경우 처리
    if num_classes < num_clusters:
        print(f"Warning: Found {num_classes} classes, but {num_clusters} prototypes needed.")
        print("Will use class means for available classes and keep random initialization for others.")
    
    # 클래스별 평균 임베딩 계산
    for i, label in enumerate(unique_labels):
        if i >= num_clusters:  # 프로토타입 수보다 많은 클래스는 무시
            break
            
        mask = (labels == label)
        class_embeddings = embeddings[mask]
    
        if len(class_embeddings) > 0:
            class_mean = F.normalize(torch.mean(class_embeddings, dim=0), dim=0)
            prototypes[i] = class_mean
    
    # 정규화된 프로토타입 반환
    return F.normalize(prototypes, dim=1)
# -------------------------------------------------------------------

@torch.no_grad()
def sinkhorn_knopp(scores, temperature, sk_iterations, device):
    """Sinkhorn-Knopp 알고리즘 (순수 PyTorch 함수 버전)"""
    Q = torch.exp(scores / temperature).t()
    Q /= torch.sum(Q)
    
    K, B = Q.shape
    r = torch.ones(K, device=device) / K
    c = torch.ones(B, device=device) / B
    
    for _ in range(sk_iterations):
        sum_Q_row = torch.sum(Q, dim=1, keepdim=True)  # 크기: (K, 1)
        Q *= (r.view(-1, 1) / sum_Q_row)  # r을 (K, 1) 크기로 변환하여 나눗셈
        
        sum_Q_col = torch.sum(Q, dim=0, keepdim=True)
        Q *= (c / sum_Q_col)
        
    return (Q / torch.sum(Q, dim=0, keepdim=True)).t()

def evaluate(model, dataloader, device, args, epoch, stage="val"):
    """모델 평가 함수 (순수 PyTorch)"""
    model.eval() # 평가 모드
    all_embs_rule, all_embs_1, all_embs_base = [], [], []
    all_labels, all_predicted_rule, all_predicted_1, all_predicted_base = [], [], [], []

    num_sensors = 97

    with torch.no_grad():
        for videos, imu_data, labels in dataloader:
            imu_data, labels = imu_data.to(device), labels.to(device)
            label_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 8}
            labels = torch.tensor([label_mapping[label.item()] for label in labels])
            labels = labels.to(device)

            # 1. 규칙 기반 특징 + 딥러닝 모델
            rule_based_feature = get_representative_sensor_feature(imu_data, labels, num_sensors, args.top_k)
            scores_rule, final_feature_rule, _ = model(imu_data, rule_based_feature)
            scores_rule_sk = sinkhorn_knopp(scores_rule, args.temperature, args.sk_iterations, device)
            predicted_rule = torch.argmax(scores_rule_sk, dim=1)

            # 2. 딥러닝 모델 Only (alpha=0.5, 훈련 중과 유사)
            rule_based_feature_zero = torch.zeros_like(rule_based_feature)
            scores_1, final_feature_1, alpha_1 = model(imu_data, rule_based_feature_zero)
            scores_1_sk = sinkhorn_knopp(scores_1, args.temperature, args.sk_iterations, device)
            predicted_1 = torch.argmax(scores_1_sk, dim=1)

            # 3. 딥러닝 모델 Only (alpha=1.0, 최종 성능)
            scores_base, final_feature_base, alpha_base = model(imu_data, rule_based_feature_zero, val=True)
            scores_base_sk = sinkhorn_knopp(scores_base, args.temperature, args.sk_iterations, device)
            predicted_base = torch.argmax(scores_base_sk, dim=1)

            # 결과 저장
            all_embs_rule.append(final_feature_rule.cpu().numpy())
            all_embs_1.append(final_feature_1.cpu().numpy())
            all_embs_base.append(final_feature_base.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predicted_rule.append(predicted_rule.cpu().numpy())
            all_predicted_1.append(predicted_1.cpu().numpy())
            all_predicted_base.append(predicted_base.cpu().numpy())

    all_embs_rule = np.vstack(all_embs_rule)
    all_embs_1 = np.vstack(all_embs_1)
    all_embs_base = np.vstack(all_embs_base)
    all_labels = np.concatenate(all_labels)
    all_predicted_rule = np.concatenate(all_predicted_rule)
    all_predicted_1 = np.concatenate(all_predicted_1)
    all_predicted_base = np.concatenate(all_predicted_base)

    # 헝가리안 매칭으로 정확도 계산
    accuracy_rule, mapping_rule = compute_hungarian_matching(all_predicted_rule, all_labels, args.num_classes)
    accuracy_1, mapping_1 = compute_hungarian_matching(all_predicted_1, all_labels, args.num_classes)
    accuracy_base, mapping_base = compute_hungarian_matching(all_predicted_base, all_labels, args.num_classes)
    
    print(f"[{stage.upper()} Epoch {epoch}] Acc (Rule): {accuracy_rule:.4f}, Acc 1(Alpha=0.5): {accuracy_1:.4f}, Acc Base(Alpha=1.0): {accuracy_base:.4f}")
    
    # wandb 로깅
    wandb.log({
        f'{stage}_acc_rule': accuracy_rule,
        f'{stage}_acc_1': accuracy_1,
        f'{stage}_acc_base': accuracy_base,
        'epoch': epoch
    })
    if type(alpha_base) == torch.Tensor:
        wandb.log({
            f'{stage}_alpha': alpha_base.mean().item(),
            'epoch': epoch
        })
    elif type(alpha_base) == float or type(alpha_base) == int:
        wandb.log({
            f'{stage}_alpha': alpha_base,
            'epoch': epoch
        })
    if type(alpha_1) == torch.Tensor:
        wandb.log({
            f'train_alpha': alpha_1.mean().item(),
            'epoch': epoch
        })
    elif type(alpha_1) == float or type(alpha_1) == int:
        wandb.log({
            f'train_alpha': alpha_1,
            'epoch': epoch
        })

    # # t-SNE 시각화 (선택적으로 짝수 에포크에만 실행)
    if epoch %2== 0:
        title = f"t-SNE at Epoch {epoch}"
        prototypes_np = model.prototypes.detach().cpu().numpy()
        all_labels_mapped = [mapping_1.get(label.item()) for label in all_labels]
        fig_rule = visualize_tsne(all_embs_rule, all_labels_mapped, all_predicted_rule, title+"_rule", prototypes=prototypes_np, num_classes=args.num_classes)
        # fig_1 = visualize_tsne(all_embs_1, all_labels, all_predicted_1, title+"_1", prototypes=prototypes_np, num_classes=args.num_classes)
        # fig_base = visualize_tsne(all_embs_base, all_labels, all_predicted_base, title+"_base", prototypes=prototypes_np, num_classes=args.num_classes)
        wandb.log({
            "t-SNE Visualization Rule": wandb.Image(fig_rule),
            # "t-SNE Visualization 1": wandb.Image(fig_1),
            # "t-SNE Visualization Base": wandb.Image(fig_base)
        })
        plt.close('all')

    model.train() # 다시 학습 모드로 전환
    return mapping_1 # 훈련 스텝에서 사용할 매핑 반환


def train_one_epoch_with_clustering_model(
    clustering_model, 
    dataloader, 
    optimizer, 
    device, 
    epoch, 
    args # temperature, sk_iterations, top_k 등 하이퍼파라미터를 담은 객체
):
    """
    하이브리드 클러스터링 모델의 한 에포크 학습을 수행합니다.

    Args:
        clustering_model (nn.Module): 학습시킬 GatedHybridModel.
        dataloader (DataLoader): 학습용 데이터 로더.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        device (torch.device): 학습을 수행할 장치 (e.g., 'cuda' or 'cpu').
        epoch (int): 현재 에포크 번호.
        args (Namespace): 하이퍼파라미터 및 설정을 담고 있는 객체.

    Returns:
        float: 해당 에포크의 평균 학습 손실.
    """
    clustering_model.train()  # 모델을 학습 모드로 설정
    clustering_model.update_epoch(epoch) # 모델의 내부 에포크 카운터 업데이트 (alpha 스케줄링용)

    total_loss = 0.0
    total_mse_loss = 0.0
    total_diversity_loss = 0.0
    
    # 데이터 로더에 tqdm을 적용하여 진행률 표시
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
    
    # dataloader에서 imu_data와 labels만 사용한다고 가정
    for imu_data, labels in pbar:
        imu_data = imu_data.to(device)
        labels = labels.to(device) # 레이블도 GPU로 이동

        # 1. 그래디언트 초기화
        optimizer.zero_grad()

        # 2. 규칙 기반 특징 추출
        num_sensors = imu_data.shape[1] # 데이터에서 센서 수 확인
        rule_based_feature = get_representative_sensor_feature(imu_data, labels, num_sensors, args.top_k)
        
        # 3. 모델 순전파 (Forward Pass)
        scores, final_feature, alpha = clustering_model(imu_data, rule_based_feature)
        
        # 4. Sinkhorn-Knopp 알고리즘으로 클러스터 할당 확률 계산
        scores_sk = sinkhorn_knopp(scores, args.temperature, args.sk_iterations, device)
        
        # 5. 손실(Loss) 계산
        with torch.no_grad():
            pseudo_labels = torch.argmax(scores_sk, dim=1)
        
        # 5-1. 클러스터링 MSE 손실
        mse_loss = F.mse_loss(final_feature, clustering_model.prototypes[pseudo_labels])

        # 5-2. 프로토타입 다양성(Diversity) 손실
        prototypes = clustering_model.prototypes
        p1 = prototypes.unsqueeze(1)
        p2 = prototypes.unsqueeze(0)
        mse_matrix = F.mse_loss(p1, p2, reduction='none').mean(dim=2)
        n_proto = args.num_classes
        # 프로토타입 간 거리가 멀어지도록(MSE가 커지도록) 손실에 음수를 취함
        diversity_loss = - (mse_matrix.sum()) / (n_proto * (n_proto - 1))
        
        # 5-3. 최종 손실
        loss = mse_loss + diversity_loss
        
        # 6. 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        # 7. 통계 기록
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_diversity_loss += diversity_loss.item()

        # 진행률 바에 현재 손실 값 표시
        pbar.set_postfix({
            "Loss": loss.item(), 
            "MSE": mse_loss.item(), 
            "Diversity": diversity_loss.item(),
            "Alpha": alpha.mean().item() if isinstance(alpha, torch.Tensor) else alpha
        })

    # 에포크 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_diversity_loss = total_diversity_loss / len(dataloader)
    
    print(f"\nEpoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f}, Avg MSE: {avg_mse_loss:.4f}, Avg Diversity: {avg_diversity_loss:.4f}")

    return avg_loss
