import os
import numpy as np
import matplotlib.pyplot as plt
from visualizes import visualize_tsne   # 네가 만든 함수 그대로 사용

# ---- 입력 폴더 & 출력 폴더 ----
input_dir = "/mnt/hdd4tb/junho/HWU-USP_v2/tsne_cache/primus_detailed_prudent"
output_dir = "/home/jaemo/z_tsne"
os.makedirs(output_dir, exist_ok=True)

print("➡️ 입력 폴더:", input_dir)
print("➡️ 출력 폴더:", output_dir)

# ---- 폴더를 순회 ----
for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".npz"):
        continue

    npz_path = os.path.join(input_dir, fname)
    print(f"\n📂 처리 중: {fname}")

    # ---- 1) NPZ 로드 ----
    data = np.load(npz_path)
    z_video = data["z_video"]
    z_sensor = data["z_sensor"]
    labels   = data["labels"]

    # ---- 2) 라벨 재구성 ----
    old_labels = labels.copy()
    new_labels = np.zeros_like(old_labels)

    # close = 0
    new_labels[np.isin(old_labels, [0,1,2,3])] = 0
    # open = 1
    new_labels[np.isin(old_labels, [4,5,6,7])] = 1

    # random(8) 제거
    mask = old_labels != 8
    
    z_video_f  = z_video[mask]
    z_sensor_f = z_sensor[mask]
    labels_f   = new_labels[mask]

    num_clusters = len(np.unique(labels_f))

    # ---- 3) t-SNE 그리기 ----

    # ⚠️ plot 이름 prefix (확장자 제거)
    prefix = fname.replace(".npz", "")

    # Z_video 시각화
    fig2d, fig3d = visualize_tsne(
        embeddings=z_video_f,
        true_labels=labels_f,
        pred_labels=None,
        prototypes=None,
        title=f"Z_video t-SNE {prefix}",
        num_classes=num_clusters,
        dataset_name="HWU-USP",
    )
    fig2d.savefig(os.path.join(output_dir, f"{prefix}_video_2d.png"), dpi=200)
    fig3d.savefig(os.path.join(output_dir, f"{prefix}_video_3d.png"), dpi=200)
    plt.close(fig2d)
    plt.close(fig3d)

    # Z_sensor 시각화
    fig2d, fig3d = visualize_tsne(
        embeddings=z_sensor_f,
        true_labels=labels_f,
        pred_labels=None,
        prototypes=None,
        title=f"Z_sensor t-SNE {prefix}",
        num_classes=num_clusters,
        dataset_name="HWU-USP",
    )
    fig2d.savefig(os.path.join(output_dir, f"{prefix}_sensor_2d.png"), dpi=200)
    fig3d.savefig(os.path.join(output_dir, f"{prefix}_sensor_3d.png"), dpi=200)
    plt.close(fig2d)
    plt.close(fig3d)

    print(f"✅ {fname} 완료")

print("\n🎉 전체 t-SNE 변환 완료!")
print(f"📁 저장 위치: {output_dir}")
