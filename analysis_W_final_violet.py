import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# =========================================================
# 🔧 기본 경로 설정
# =========================================================
prefix = ""
base_dir = f"/mnt/hdd4tb/junho/Opportunity++/{prefix}weights"
save_root = f"/home/jaemo/Method/checkpoints/method/{prefix}qualitative"
# [수정] 저장 폴더 이름 변경
save_dir = os.path.join(save_root, "violin_plots_class_top_bottom_6")
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# 🎨 유틸 및 설정
# =========================================================
class_names = {
    0: 'Open Door 1', 1: 'Open Door 2',
    2: 'Close Door 1', 3: 'Close Door 2',
    4: 'Open Fridge', 5: 'Close Fridge',
    6: 'Open Dishwasher', 7: 'Close Dishwasher',
    8: 'Open Drawer 1', 9: 'Close Drawer 1',
    10:'Open Drawer 2', 11:'Close Drawer 2',
    12:'Open Drawer 3', 13:'Close Drawer 3'
}
# [NEW] 클래스 ID -> 객체 이름 매핑
class_to_object = {
    0: "Door1", 1: "Door2", 2: "Door1", 3: "Door2",
    4: "Fridge", 5: "Fridge", 6: "Dishwasher", 7: "Dishwasher",
    8: "Drawer1", 9: "Drawer1", 10: "Drawer2", 11: "Drawer2",
    12: "Drawer3", 13: "Drawer3"
}
COLOR_MAP = {"False": "skyblue", "Hard": "salmon"}
epoch_delta_scores = {}

# [NEW] 클래스 단위 델타 계산
def get_class_delta(Wf, labels_np, false_mask, hard_mask, class_id):
    anchor_mask_2d = (labels_np[:, None] == class_id)
    false_vals = Wf[false_mask & anchor_mask_2d]
    hard_vals  = Wf[hard_mask  & anchor_mask_2d]
    
    if len(false_vals) == 0 or len(hard_vals) == 0:
        return -np.inf
    return np.mean(hard_vals) - np.mean(false_vals)

# =========================================================
# 🔁 메인 루프: 에포크 파일 순회
# =========================================================
pkl_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".pkl")])
print(f"📂 Found {len(pkl_files)} epoch files. Start processing Class-wise Top/Bottom 3 Violin Plots...")

for fname in tqdm(pkl_files):
    epoch_id = fname.replace("epoch_", "").replace("_stepwise.pkl", "")
    pkl_path = os.path.join(base_dir, fname)

    try:
        with open(pkl_path, "rb") as f: data = pickle.load(f)
    except Exception as e: print(f"⚠️ Failed to load {fname}: {e}"); continue
    if len(data) == 0: continue

    step = data[0]
    required_keys = ["W_final", "labels_np", "spatial_labels_np", "motion_labels_np"]
    if not all(k in step for k in required_keys):
        print(f"⚠️ Missing keys in {fname}, skipping.")
        continue

    # 1. 데이터 준비 및 마스크 생성
    Wf = step["W_final"]
    labels_np = step["labels_np"]
    spatial = step["spatial_labels_np"]
    motion  = step["motion_labels_np"]
    N = len(labels_np)
    mask_upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    same_sp = (spatial[:, None] == spatial[None, :])
    same_mo = (motion[:, None]  == motion[None, :])
    false_mask = (same_sp & same_mo) & mask_upper
    hard_mask  = (same_sp & ~same_mo) & mask_upper
    
    # 2. [수정] 14개 '클래스' 전체에 대해 델타 계산
    class_deltas = {}
    for class_idx, class_name in class_names.items():
        delta = get_class_delta(Wf, labels_np, false_mask, hard_mask, class_idx)
        class_deltas[class_name] = delta
            
    if not class_deltas: continue

    # 3. [수정] Delta 기준 정렬 및 Top/Bottom 3 '클래스' 선별 (겹침 방지)
    valid_deltas = {k: v for k, v in class_deltas.items() if v > -np.inf}
    if len(valid_deltas) < 6: continue
        
    sorted_classes = sorted(valid_deltas.items(), key=lambda item: item[1])
    
    # Top 3 클래스 선별
    top_3_pairs = sorted_classes[-3:]
    top_3_names = [c[0] for c in top_3_pairs][::-1] # 델타 큰 순
    
    # Top 3에 속한 '객체'들 파악
    top_3_objects = set(class_to_object[k] for k, v in class_names.items() if v in top_3_names)
    
    # Top 3 객체를 제외한 나머지 클래스들
    remaining_classes = [c for c in sorted_classes[:-3] if class_to_object[ [k for k,v in class_names.items() if v == c[0]][0] ] not in top_3_objects]
    
    # 남은 것들 중에서 Bottom 3 클래스 선별
    bottom_3_pairs = remaining_classes[:3]
    bottom_3_names = [c[0] for c in bottom_3_pairs] # 델타 작은 순
    
    selected_class_names = top_3_names + bottom_3_names # 총 6개 클래스
    
    # Best Epoch 점수 기록 (Top-3 클래스의 평균 델타)
    epoch_delta_scores[epoch_id] = np.mean([p[1] for p in top_3_pairs])

    # 4. Top/Bottom 6개 '클래스'에 대해서만 데이터 수집
    plot_data = []
    for class_name in selected_class_names:
        class_idx = [k for k, v in class_names.items() if v == class_name][0]
        anchor_mask_2d = (labels_np[:, None] == class_idx)
        
        false_vals = Wf[false_mask & anchor_mask_2d]
        hard_vals  = Wf[hard_mask  & anchor_mask_2d]

        for val in false_vals: plot_data.append([class_name, "False", val])
        for val in hard_vals:  plot_data.append([class_name, "Hard", val])
            
    if not plot_data: continue
        
    df_plot = pd.DataFrame(plot_data, columns=["Class", "Type", "W_final"])

    # 5. Violin Plot 생성 (6개 클래스 + 구분선)
    plt.figure(figsize=(14, 7)) # 6개 클래스에 맞게 폭 조절
    sns.violinplot(
        data=df_plot[df_plot['Type'].isin(['False', 'Hard'])],
        x="Class",
        y="W_final",
        hue="Type",
        split=True,
        inner="quartile",
        palette=COLOR_MAP,
        hue_order=["False", "Hard"],
        order=selected_class_names # Top-3, Bottom-3 순서
    )
    
    plt.axvline(x=2.5, color='black', linestyle='--', linewidth=2, label='Top-3 Classes vs Bottom-3 Classes')
    
    plt.title(f"Epoch {epoch_id} — W_final Distribution (Top-3 vs Bottom-3 Classes)", fontsize=16)
    plt.ylabel("W_final Value")
    plt.xlabel("Class (Sorted by Class-level Δ, Object-Disjoint)")
    plt.xticks(rotation=15, ha='right') 
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"epoch_{epoch_id}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()

print(f"\n🎯 All Class-wise Top/Bottom 3 Violin plots processed successfully!")

# =========================================================
# 6. Best Epoch 선정 및 출력 (Top-3 클래스 평균 델타 기준)
# =========================================================
if epoch_delta_scores:
    best_epoch_id = max(epoch_delta_scores, key=lambda k: epoch_delta_scores[k])
    best_score = epoch_delta_scores[best_epoch_id]
    
    print("\n" + "="*40)
    print(f"🏆 Best Epoch Analysis (Max Top-3 Class Delta)")
    print(f"Best Epoch: {best_epoch_id}")
    print(f"Score (Top-3 Avg Delta): {best_score:.4f}")
    print("="*40)