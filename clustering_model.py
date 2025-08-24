import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# --- 사용자 정의 모듈 및 헬퍼 함수 (기존 코드와 동일) ---

START_INDEX = 134
END_INDEX = 231

ACTION_MERGE_LABELS = {
        0: 'Door 1',
        1: 'Door 2',
        2: 'Fridge',
        3: 'Dishwasher',
        4: 'Drawer 1',
        5: 'Drawer 2',
        6: 'Drawer 3',
        7: 'Clean Table',
        8: 'Drink from Cup',
        9: 'Toggle Switch'
    }

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
    """
    sensor_index에 해당하는 센서 이름을 반환합니다.
    
    Args:
        sensor_index: 센서 인덱스 (1-based index)
    
    Returns:
        str: 센서 이름 문자열, 해당 인덱스가 없으면 "Unknown Sensor"
    """
    # 파일 경로 설정
    column_names_path = "/mnt/hdd4tb/junho/Opportunity++/data/column_names.txt"
    
    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(column_names_path):
            return f"Unknown Sensor (Index: {sensor_index})"
        
        # 파일 읽기
        with open(column_names_path, 'r') as f:
            lines = f.readlines()
        
        # 지정된 인덱스 찾기
        for line in lines:
            # Column: {index} {description} 형식 찾기
            if line.strip().startswith(f"Column: {sensor_index} "):
                # 센서 설명 추출
                sensor_description = line.strip()[len(f"Column: {sensor_index} "):]
                
                # 센서 이름과 타입 파싱 (예: "Accelerometer RKN^ accX")
                parts = sensor_description.split(';')[0].strip().split()
                if len(parts) >= 2:
                    sensor_type = parts[0]  # "Accelerometer"
                    sensor_location = parts[1]  # "RKN^"
                    sensor_axis = " ".join(parts[2:])  # "accX"
                    return f"{sensor_type} {sensor_location} {sensor_axis}"
                else:
                    return sensor_description
                
        # 인덱스가 없으면
        return f"Unknown Sensor (Index: {sensor_index})"
    
    except Exception as e:
        return f"Error reading sensor name: {str(e)}"

class ClusteringModel(nn.Module):
    def __init__(self, encoder, embedding_dim, num_sensors, num_clusters, prototypes=None, alpha_fixed=False, total_epochs=20, threshold_epoch=6):
        super().__init__()
        self.encoder = encoder
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

    def get_representative_sensor_feature(self, imu_batch, labels, num_total_sensors=10, top_k=1):
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

    def compute_hungarian_matching(self, pred_labels, true_labels, num_clusters):
        cost_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int64)
        for i in range(len(pred_labels)):
            cost_matrix[pred_labels[i], true_labels[i]] += 1
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        mapping = {i: j for i, j in zip(row_ind, col_ind)}
        return accuracy_score(true_labels, np.array([mapping.get(x, -1) for x in pred_labels])), mapping


    @torch.no_grad()
    def sinkhorn_knopp(self, scores, temperature, sk_iterations, device):
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

    def evaluate(self, dataloader, device, args, epoch, stage="val"):
        """모델 평가 함수 (순수 PyTorch)"""
        self.eval() # 평가 모드
        all_embs_rule, all_embs_1, all_embs_base = [], [], []
        all_labels, all_predicted_rule, all_predicted_1, all_predicted_base = [], [], [], []
        all_item_id = []
        num_sensors = 97

        with torch.no_grad():
            for videos, imu_data, labels, item_id in dataloader:
                imu_data, labels = imu_data.to(device), labels.to(device)
                label_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 8}
                labels = torch.tensor([label_mapping[label.item()] for label in labels])
                labels = labels.to(device)

                # 1. 규칙 기반 특징 + 딥러닝 모델
                rule_based_feature = get_representative_sensor_feature(imu_data, labels, num_sensors, args.top_k)

                # rule only 추측 모델
                scores_rule, final_feature_rule, _ = self(imu_data, rule_based_feature)
                scores_rule_sk = self.sinkhorn_knopp(scores_rule, args.temperature, args.sk_iterations, device)
                predicted_rule = torch.argmax(scores_rule_sk, dim=1)

                # 2. 딥러닝 모델 Only (alpha=0.5, 훈련 중과 유사)
                rule_based_feature_zero = torch.zeros_like(rule_based_feature)
                scores_1, final_feature_1, alpha_1 = self(imu_data, rule_based_feature_zero)
                scores_1_sk = self.sinkhorn_knopp(scores_1, args.temperature, args.sk_iterations, device)
                predicted_1 = torch.argmax(scores_1_sk, dim=1)

                # 3. 딥러닝 모델 Only (alpha=1.0, 최종 성능)
                scores_base, final_feature_base, alpha_base = self(imu_data, rule_based_feature_zero, val=True)
                scores_base_sk = self.sinkhorn_knopp(scores_base, args.temperature, args.sk_iterations, device)
                predicted_base = torch.argmax(scores_base_sk, dim=1)

                # 결과 저장
                all_embs_rule.append(final_feature_rule.cpu().numpy())
                all_embs_1.append(final_feature_1.cpu().numpy())
                all_embs_base.append(final_feature_base.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_predicted_rule.append(predicted_rule.cpu().numpy())
                all_predicted_1.append(predicted_1.cpu().numpy())
                all_predicted_base.append(predicted_base.cpu().numpy())
                all_item_id.append(item_id)
        all_embs_rule = np.vstack(all_embs_rule)
        all_embs_1 = np.vstack(all_embs_1)
        all_embs_base = np.vstack(all_embs_base)
        all_labels = np.concatenate(all_labels)
        all_predicted_rule = np.concatenate(all_predicted_rule)
        all_predicted_1 = np.concatenate(all_predicted_1)
        all_predicted_base = np.concatenate(all_predicted_base)
        all_item_id = np.concatenate(all_item_id)
        # 헝가리안 매칭으로 정확도 계산
        accuracy_rule, mapping_rule = self.compute_hungarian_matching(all_predicted_rule, all_labels, args.num_classes)
        accuracy_1, mapping_1 = self.compute_hungarian_matching(all_predicted_1, all_labels, args.num_classes)
        accuracy_base, mapping_base = self.compute_hungarian_matching(all_predicted_base, all_labels, args.num_classes)
        
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
            prototypes_np = self.prototypes.detach().cpu().numpy()
            all_predicted_rule_mapped = [mapping_rule.get(label.item()) for label in all_predicted_rule]
            fig_rule = visualize_tsne(all_embs_rule, all_labels, all_predicted_rule_mapped, title+"_rule", prototypes=prototypes_np, num_classes=args.num_classes)
            # 틀린 놈 찾기
            for i in range(len(all_predicted_rule_mapped)):
                if all_labels[i] != all_predicted_rule_mapped[i]:
                    print(f"Epoch {epoch} Rule: {all_item_id[i]}")
                    print(f"Epoch {epoch} Rule: {all_labels[i]} -> {all_predicted_rule_mapped[i]}")
                    print("--------------------------------")
            # fig_1 = visualize_tsne(all_embs_1, all_labels, all_predicted_1, title+"_1", prototypes=prototypes_np, num_classes=args.num_classes)
            # fig_base = visualize_tsne(all_embs_base, all_labels, all_predicted_base, title+"_base", prototypes=prototypes_np, num_classes=args.num_classes)
            wandb.log({
                "t-SNE Visualization Rule": wandb.Image(fig_rule),
                # "t-SNE Visualization 1": wandb.Image(fig_1),
                # "t-SNE Visualization Base": wandb.Image(fig_base)
            })
            plt.close('all')

        self.train() # 다시 학습 모드로 전환
        return mapping_1 # 훈련 스텝에서 사용할 매핑 반환

# def initialize_prototypes(dataloader, num_clusters, embedding_dim, num_sensors, device, args):
#     """첫 번째 배치에서 프로토타입 초기화"""
#     # 임시 인코더 생성
#     temp_encoder = MW2StackRNNPooling(size_embeddings=embedding_dim, in_channels=num_sensors).to(device)
#     temp_projection = nn.Linear(num_sensors, embedding_dim).to(device)
    
#     # 임베딩 수집
#     embeddings = []
#     labels = []
    
#     with torch.no_grad():
#         # 첫 배치만 사용
#         for batch_idx, batch in enumerate(dataloader):
#             if batch_idx >= 40:  # 첫 배치만 사용
#                 break
                
#             imu_data = batch[1].to(device)
#             batch_labels = batch[2].to(device)
#             label_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 8}
#             batch_labels = torch.tensor([label_mapping[label.item()] for label in batch_labels])
#             batch_labels = batch_labels.to(device)
            
#             # 규칙 기반 특징 추출
#             rule_based_feature = self.get_representative_sensor_feature(imu_data, batch_labels, num_total_sensors=num_sensors, top_k=args.top_k)
            
#             # 딥러닝 임베딩 추출
#             deep_embedding = temp_encoder(imu_data)['emb']
            
#             # 규칙 기반 특징 투영
#             rule_based_feature = temp_projection(rule_based_feature)
            
#             # 최종 임베딩 계산 (alpha=0.5 사용)
#             final_feature = 0.5 * F.normalize(deep_embedding, dim=1) + 0.5 * rule_based_feature
            
#             embeddings.append(final_feature)
#             labels.append(batch_labels)
    
#     if not embeddings:  # 배치가 비어있을 경우 랜덤 초기화
#         print("Warning: Empty batch, using random initialization")
#         return torch.randn(num_clusters, embedding_dim)
        
#     embeddings = torch.cat(embeddings, dim=0)
#     labels = torch.cat(labels, dim=0)
    
#     # 클래스별 평균 임베딩을 프로토타입으로 사용
#     unique_labels = torch.unique(labels)
#     num_classes = len(unique_labels)
    
#     # 초기 프로토타입 랜덤 초기화
#     prototypes = torch.randn(num_clusters, embedding_dim)
    
#     # 클래스 수가 프로토타입 수와 다를 경우 처리
#     if num_classes < num_clusters:
#         print(f"Warning: Found {num_classes} classes, but {num_clusters} prototypes needed.")
#         print("Will use class means for available classes and keep random initialization for others.")
    
#     # 클래스별 평균 임베딩 계산
#     for i, label in enumerate(unique_labels):
#         if i >= num_clusters:  # 프로토타입 수보다 많은 클래스는 무시
#             break
            
#         mask = (labels == label)
#         class_embeddings = embeddings[mask]
    
#         if len(class_embeddings) > 0:
#             class_mean = F.normalize(torch.mean(class_embeddings, dim=0), dim=0)
#             prototypes[i] = class_mean
    
#     # 정규화된 프로토타입 반환
#     return F.normalize(prototypes, dim=1)
# # -------------------------------------------------------------------