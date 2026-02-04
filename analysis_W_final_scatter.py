import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================================
# 🔧 기본 경로 설정
# =========================================================
# prefix = "normed_"
prefix = ""
base_dir = f"/mnt/hdd4tb/junho/Opportunity++/{prefix}weights"  # 저장된 pkl 경로
save_root = f"/home/jaemo/Method/checkpoints/method/{prefix}qualitative"

os.makedirs(os.path.join(save_root, "cdf"), exist_ok=True)
os.makedirs(os.path.join(save_root, "scatter"), exist_ok=True)

# =========================================================
# 🎨 유틸 함수
# =========================================================
def plot_cdf(values, label, color):
    sorted_vals = np.sort(values)
    cdf = np.arange(len(sorted_vals)) / len(sorted_vals)
    plt.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)

# =========================================================
# 🔁 epoch 파일 순회
# =========================================================
for fname in sorted(os.listdir(base_dir)):
    if not fname.endswith(".pkl"):
        continue

    epoch_id = fname.replace("epoch_", "").replace("_stepwise.pkl", "")
    save_cdf_path = os.path.join(save_root, "cdf", f"epoch_{epoch_id}.png")
    save_scatter_path = os.path.join(save_root, "scatter", f"epoch_{epoch_id}.png")

    # 이미 결과가 있으면 스킵
    # if os.path.exists(save_cdf_path) and os.path.exists(save_scatter_path):
    #     print(f"✅ Skipping epoch {epoch_id} (already processed)")
    #     continue

    full_path = os.path.join(base_dir, fname)
    try:
        with open(full_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load {fname}: {e}")
        continue

    if len(data) == 0:
        continue

    # =========================================================
    # 1️⃣ 첫 step만 대표로 시각화 (원하면 평균화도 가능)
    # =========================================================
    step = data[0]
    if "W_final" not in step or "spatial_labels_np" not in step:
        print(f"⚠️ Missing keys in {fname}, skipping.")
        continue

    W_final = step["W_final"]
    spatial_labels = step["spatial_labels_np"]
    motion_labels = step["motion_labels_np"]

    N = len(spatial_labels)
    mask_upper = np.triu(np.ones((N, N), dtype=bool), k=1)

    same_spatial = spatial_labels[:, None] == spatial_labels[None, :]
    same_motion  = motion_labels[:, None] == motion_labels[None, :]

    false_mask = (same_spatial & same_motion) & mask_upper
    hard_mask  = (same_spatial & ~same_motion) & mask_upper
    easy_mask  = (~same_spatial & same_motion) & mask_upper

    # =========================================================
    # 2️⃣ CDF Plot — Per Class
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

    labels_np = step["labels_np"]
    unique_classes = np.unique(labels_np)

    # 전체 sample × sample 배열
    Wf = W_final

    # for cls in unique_classes:
    #     # (1) 폴더 생성
    #     cls_name = class_names[int(cls)]
    #     cls_folder = cls_name.replace(" ", "_")  # e.g. Open Door 1 → Open_Door_1

    #     cls_dir = os.path.join(save_root, "cdf", cls_folder)
    #     os.makedirs(cls_dir, exist_ok=True)

    #     # (2) anchor가 해당 클래스인 index
    #     anchor_mask = (labels_np == cls)

    #     if not np.any(anchor_mask):
    #         continue

    #     # (3) pair filtering (upper triangle)
    #     # anchor가 포함된 row만 선택
    #     class_pairs = (anchor_mask[:, None] | anchor_mask[None, :]) & mask_upper

    #     cdf_false = Wf[false_mask & class_pairs]
    #     cdf_hard  = Wf[hard_mask  & class_pairs]
    #     cdf_easy  = Wf[easy_mask  & class_pairs]

    #     # (4) plot
    #     plt.figure(figsize=(6, 4))
    #     if len(cdf_false) > 0: plot_cdf(cdf_false, "False", "skyblue")
    #     if len(cdf_hard)  > 0: plot_cdf(cdf_hard,  "Hard",  "salmon")
    #     if len(cdf_easy)  > 0: plot_cdf(cdf_easy,  "Easy",  "lightgreen")

    #     plt.xlabel("W_final value")
    #     plt.ylabel("Cumulative probability")
    #     plt.title(f"Epoch {epoch_id} - CDF for {cls_name}")
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #     plt.tight_layout()

    #     save_cdf_path = os.path.join(cls_dir, f"epoch_{epoch_id}.png")
    #     plt.savefig(save_cdf_path)
    #     plt.close()

    #     print(f"📌 Saved CDF → {save_cdf_path}")

    # 3️⃣ Scatter Plot (Spatial vs Temporal) — Per-Class
    # =========================================================
    sim_vid_app = step.get("sim_app_vid")
    sim_sen_app = step.get("sim_app_sen")

    if sim_vid_app is None or sim_sen_app is None:
        print(f"⚠️ Missing sim_app/sim_temp in {fname}, skipping scatter.")
        continue

    sim_app = np.maximum(sim_vid_app, sim_sen_app)

    sim_vid_temp = step.get("sim_vid_mom")
    sim_sen_temp = step.get("sim_sen_mom")
    sim_temp = (sim_vid_temp + sim_sen_temp) / 2

    app_vals = sim_app.flatten()
    temp_vals = sim_temp.flatten()

    # pair-type label
    rel = np.full(len(app_vals), "None", dtype=object)
    rel[false_mask.flatten()] = "False Negatives"
    rel[hard_mask.flatten()] = "Hard Negatives"
    rel[easy_mask.flatten()] = "Easy Negatives"

    # =============================
    # 🔥 class name mapping
    # =============================
    class_names = {
        0: 'Open Door 1',     1: 'Open Door 2',
        2: 'Close Door 1',    3: 'Close Door 2',
        4: 'Open Fridge',     5: 'Close Fridge',
        6: 'Open Dishwasher', 7: 'Close Dishwasher',
        8: 'Open Drawer 1',   9: 'Close Drawer 1',
        10:'Open Drawer 2',   11:'Close Drawer 2',
        12:'Open Drawer 3',   13:'Close Drawer 3'
    }

    labels_np = step["labels_np"]
    unique_classes = np.unique(labels_np)

    for cls in unique_classes:
        class_mask = (labels_np == cls)
        if not np.any(class_mask):
            continue

        # mask for pair related to this class (flattened)
        class_pair_mask = (class_mask[:, None] | class_mask[None, :]).flatten()

        # subset extraction
        sub_app  = app_vals[class_pair_mask]
        sub_temp = temp_vals[class_pair_mask]
        sub_rel  = rel[class_pair_mask]

        # ======== Folder name =========
        cls_name = class_names[int(cls)]
        cls_folder = cls_name.replace(" ", "_")  # e.g. Open Door 1 → Open_Door_1

        cls_dir = os.path.join(save_root, "scatter", cls_folder)
        os.makedirs(cls_dir, exist_ok=True)

        save_scatter_path = os.path.join(cls_dir, f"epoch_{epoch_id}.png")

        # ======== Scatter plot =========
        plt.figure(figsize=(6, 6))
        for k, color in {"Easy Negatives": "lightgreen", "Hard Negatives": "salmon", "False Negatives": "skyblue"}.items():
            mask = (sub_rel == k)
            if np.any(mask):
                plt.scatter(sub_app[mask], sub_temp[mask], s=6, alpha=0.55, color=color, label=k)

        plt.xlabel("Spatial Similarity")
        plt.ylabel("Temporal Similarity")
        plt.title(f"Epoch {epoch_id} — {cls_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_scatter_path)
        plt.close()

        print(f"📌 Saved scatter → {save_scatter_path}")



print("\n🎯 All epochs processed!")
