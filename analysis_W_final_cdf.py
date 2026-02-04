import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# =========================================================
# 🔧 기본 경로 설정
# =========================================================
prefix = "normed_"
base_dir = f"/mnt/hdd4tb/junho/Opportunity++/{prefix}weights"
save_root = f"/home/jaemo/Method/checkpoints/method/{prefix}qualitative"

# 저장 폴더 생성
os.makedirs(os.path.join(save_root, "representative_cdf"), exist_ok=True)
os.makedirs(os.path.join(save_root, "cdf_no_legends"), exist_ok=True)
os.makedirs(os.path.join(save_root, "scatter"), exist_ok=True)

# =========================================================
# 🎨 유틸 및 설정
# =========================================================
class_names = {
    0: 'Open Door 1',     1: 'Open Door 2',
    2: 'Close Door 1',    3: 'Close Door 2',
    4: 'Open Fridge',     5: 'Close Fridge',
    6: 'Open Dishwasher', 7: 'Close Dishwasher',
    8: 'Open Drawer 1',   9: 'Close Drawer 1',
    10:'Open Drawer 2',   11:'Close Drawer 2',
    12:'Open Drawer 3',   13:'Close Drawer 3'
}

object_pairs = {
    "Door1": [0, 2],
    "Door2": [1, 3],
    "Fridge": [4, 5],
    "Dishwasher": [6, 7],
    "Drawer1": [8, 9],
    "Drawer2": [10, 11],
    "Drawer3": [12, 13],
}

# def plot_cdf(values, label, color, ax):
#     """CDF 그리기 헬퍼 함수"""
#     sorted_vals = np.sort(values)
#     cdf = np.arange(len(sorted_vals)) / len(sorted_vals)
#     ax.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
# (기존)
# def plot_cdf(values, label, color, ax):
#     """CDF 그리기 헬퍼 함수"""
#     sorted_vals = np.sort(values)
#     cdf = np.arange(len(sorted_vals)) / len(sorted_vals)
#     ax.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)

# (수정)
def plot_cdf(values, label, color, ax):
    """(수정) Seaborn KDE를 사용하여 매끄러운 CDF 그리기"""
    sns.kdeplot(
        values, 
        label=label, 
        color=color, 
        ax=ax,
        cumulative=True,  # <-- 이 옵션이 CDF로 만듭니다.
        linewidth=2,
        bw_adjust=0.5     # <-- 이 값을 조절해 부드러움 정도를 변경 (작을수록 뾰족, 클수록 뭉툭)
    )

def evaluate_class_delta(Wf, labels_np, false_mask, hard_mask, class_ids):
    """
    두 클래스(Open/Close) 중 False와 Hard의 차이가 더 큰 클래스를 선택
    """
    deltas = {}
    for cid in class_ids:
        # 해당 클래스를 앵커로 하는 마스크 (N, N)
        anchor_mask_2d = (labels_np[:, None] == cid) # Broadcasting -> (N, N)
        
        # 해당 클래스에 속하는 pair들의 가중치 값만 추출
        false_vals = Wf[false_mask & anchor_mask_2d]
        hard_vals  = Wf[hard_mask  & anchor_mask_2d]

        if len(false_vals) == 0 or len(hard_vals) == 0:
            deltas[cid] = -np.inf
        else:
            # Hard(높음) - False(낮음) 차이가 클수록 좋음
            deltas[cid] = np.mean(hard_vals) - np.mean(false_vals)

    # Delta가 가장 큰 클래스 선택
    best_class = max(deltas, key=lambda x: deltas[x])
    return best_class

# =========================================================
# 🔁 메인 루프: 에포크 파일 순회
# =========================================================
pkl_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".pkl")])

print(f"📂 Found {len(pkl_files)} epoch files. Start processing...")

for fname in tqdm(pkl_files):
    epoch_id = fname.replace("epoch_", "").replace("_stepwise.pkl", "")
    pkl_path = os.path.join(base_dir, fname)

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load {fname}: {e}")
        continue

    if len(data) == 0:
        continue

    # 첫 번째 스텝 데이터만 사용 (대표값)
    step = data[0]
    
    # 필수 키 확인
    required_keys = ["W_final", "labels_np", "spatial_labels_np", "motion_labels_np", "sim_app_vid", "sim_vid_mom"]
    if not all(k in step for k in required_keys):
        print(f"⚠️ Missing keys in {fname}, skipping.")
        continue

    # -------------------------------------------------
    # 1. 데이터 준비
    # -------------------------------------------------
    Wf = step["W_final"]
    labels_np = step["labels_np"]
    spatial = step["spatial_labels_np"]
    motion  = step["motion_labels_np"]
    N = len(labels_np)

    # 마스크 생성 (상삼각 행렬만 사용)
    mask_upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    same_sp = spatial[:, None] == spatial[None, :]
    same_mo = motion[:, None]  == motion[None, :]

    false_mask = (same_sp & same_mo) & mask_upper
    hard_mask  = (same_sp & ~same_mo) & mask_upper
    easy_mask  = (~same_sp & same_mo) & mask_upper  # Motion Confused (Easy but same motion)
    
    # 유사도 (Scatter용)
    sim_app  = np.maximum(step["sim_app_vid"], step["sim_app_sen"])
    sim_temp = (step["sim_vid_mom"] + step["sim_sen_mom"]) / 2

    # -------------------------------------------------
    # 2. 대표 클래스 선정 및 통합 데이터 수집 (CDF용)
    # -------------------------------------------------
    chosen_classes = []
    for obj, cls_pair in object_pairs.items():
        best_cls = evaluate_class_delta(Wf, labels_np, false_mask, hard_mask, cls_pair)
        chosen_classes.append(best_cls)
    
    # 선택된 대표 클래스들의 데이터만 모음
    rep_false, rep_hard, rep_easy = [], [], []
    
    for cid in chosen_classes:
        anchor_mask_2d = (labels_np[:, None] == cid)
        
        rep_false.extend(Wf[false_mask & anchor_mask_2d].tolist())
        rep_hard.extend(Wf[hard_mask & anchor_mask_2d].tolist())
        rep_easy.extend(Wf[easy_mask & anchor_mask_2d].tolist())

    # -------------------------------------------------
    # 3. Representative CDF Plot 저장
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if rep_false: plot_cdf(rep_false, "False Negatives", "blue", ax)
    if rep_hard:  plot_cdf(rep_hard,  "Hard Negatives",  "red", ax)
    if rep_easy:  plot_cdf(rep_easy,  "Easy Negatives",  "green", ax) # 필요시 추가

    ax.set_xlabel("W_final")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"Epoch {epoch_id} — Representative CDF")
    ax.grid(alpha=0.3)
    # 범례 핸들 저장
    handles, labels = ax.get_legend_handles_labels()
    
    # [수정] 그림에는 범례를 그리지 않음
    # ax.legend() # -> 이 줄을 제거!
    plt.tight_layout()
    
    # (A) 범례 없는 그림 저장

    save_path = os.path.join(save_root, "cdf_no_legends", f"epoch_{epoch_id}.png")
    plt.savefig(save_path)

    plt.close(fig)

    # --- (2) 범례만 따로 저장하기 ---
    if handles: # 범례가 있을 경우에만 실행
        # 새 Figure를 만들고 범례만 그림
        fig_legend = plt.figure(figsize=(2, 3)) # 범례 크기에 맞게 조절
        ax_legend = fig_legend.add_subplot(111)
        
        # 범례 생성
        leg = ax_legend.legend(handles, labels, loc='center', frameon=False)
        
        # 축과 배경 투명하게
        ax_legend.axis('off')
        
        # 범례만 꽉 차게 저장
        fig_legend.savefig(
            "legend_only.png",
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=True # 배경 투명하게
        )
        plt.close(fig_legend)

    print("범례 없는 그림(plot_no_legend.png)과 범례만(legend_only.png) 저장 완료!")


    # -------------------------------------------------
    # 4. Scatter Plot 저장 (Per-Class)
    # -------------------------------------------------
    # 모든 클래스에 대해 그리면 너무 많으니, 대표 클래스만 그릴 수도 있음
    # 여기서는 모든 클래스 폴더별 저장 유지
    
    unique_classes = np.unique(labels_np)
    app_vals = sim_app.flatten()
    temp_vals = sim_temp.flatten()
    
    # 관계 배열 생성 (전체 N*N)
    rel_types = np.full((N, N), "None", dtype=object)
    rel_types[false_mask] = "False Negatives"
    rel_types[hard_mask]  = "Hard Negatives"
    rel_types[easy_mask]  = "Easy Negatives"
    rel_flat = rel_types.flatten()

    for cls in unique_classes:
        cls_name = class_names[int(cls)]
        cls_folder = cls_name.replace(" ", "_")
        
        # 해당 클래스가 앵커인 행만 선택
        cls_mask_1d = (labels_np == cls)
        if not np.any(cls_mask_1d): continue
        
        # (N, N) 마스크 -> flatten
        cls_pair_mask = (cls_mask_1d[:, None] & np.ones((N, N), dtype=bool)).flatten()
        
        # 데이터 추출
        sub_app  = app_vals[cls_pair_mask]
        sub_temp = temp_vals[cls_pair_mask]
        sub_rel  = rel_flat[cls_pair_mask]
        
        # 의미 있는 관계만 필터링 (None 제외)
        valid_idx = sub_rel != "None"
        sub_app  = sub_app[valid_idx]
        sub_temp = sub_temp[valid_idx]
        sub_rel  = sub_rel[valid_idx]

        if len(sub_app) == 0: continue

        # 폴더 생성 및 저장
        cls_dir = os.path.join(save_root, "scatter", cls_folder)
        os.makedirs(cls_dir, exist_ok=True)
        
        plt.figure(figsize=(6, 6))
        
        # 순서: Easy -> Hard -> False (겹칠 때 중요한 게 위에 오도록)
        for k, color, alpha in [
            ("Easy Negatives", "lightgreen", 0.1), 
            ("Hard Negatives", "salmon", 0.6), 
            ("False Negatives", "skyblue", 0.8)
        ]:
            mask = (sub_rel == k)
            if np.any(mask):
                plt.scatter(sub_app[mask], sub_temp[mask], s=15, alpha=alpha, color=color, label=k, edgecolors='none')

        plt.xlabel("Spatial similarity")
        plt.ylabel("Temporal similarity")
        plt.title(f"Epoch {epoch_id} — {cls_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(-0.1, 1.1) # 범위 고정 (비교 용이)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        
        plt.savefig(os.path.join(cls_dir, f"epoch_{epoch_id}.png"))
        plt.close()

print("\n🎯 All epochs processed successfully!")