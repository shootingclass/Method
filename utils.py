import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


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


# --- 헝가리안 매칭을 통한 클러스터-라벨 매핑 ---
def compute_hungarian_matching(pred_labels, true_labels, num_clusters):
    """클러스터 ID와 실제 레이블 간의 최적 매핑을 찾아 정확도를 계산"""
    cost_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for i in range(len(pred_labels)):
        cost_matrix[pred_labels[i], true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapped_preds = np.zeros_like(pred_labels)
    mapping = {i: j for i, j in zip(row_ind, col_ind)}
    for i, j in mapping.items():
        mapped_preds[pred_labels == i] = j
    accuracy = np.mean(mapped_preds == true_labels)
    print("Accuracy: ", accuracy, "Mapping: ", mapping)
    return accuracy, mapping


