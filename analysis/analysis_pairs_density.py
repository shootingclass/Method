import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # seaborn 추가

# (기존 경로 설정 및 collect_pair_similarities 함수는 동일하게 유지됩니다.)
input_dir = "/mnt/hdd4tb/junho/HWU-USP_v2/tsne_cache/primus_detailed_prudent"
output_dir = "/home/jaemo/z_tsne/boxplots_epoch_debug_density" # output_dir은 그대로 사용
os.makedirs(output_dir, exist_ok=True)

def collect_pair_similarities(embeddings, labels):
    """
    embeddings: (N, D)
    labels: (N,)
      - 0~3: close
      - 4~7: open
      - 8: random -> 제거

    반환:
      spatial_sims: Same Object, Different Motion (cosine similarity)
      temporal_sims: Different Object, Same Motion (cosine similarity)
    """
    # random(8) 제거
    mask = labels != 8
    emb = embeddings[mask]
    lab = labels[mask]

    print("  ▶ unique labels (after removing 8):", np.unique(lab))

    # 라벨 구조: 0~3 close, 4~7 open
    object_id = lab % 4                          # 0,1,2,3
    motion_id = (lab >= 4).astype(np.int64)      # 0(close), 1(open)

    # --- 미리 normalize 해서 cosine similarity 빠르게 계산 ---
    # emb_norm[i] · emb_norm[j] = cosine_similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_norm = emb / norms

    spatial = []
    temporal = []

    # 1) Same Object, Different Motion
    for o in range(4):
        idx_close = np.where((object_id == o) & (motion_id == 0))[0]
        idx_open  = np.where((object_id == o) & (motion_id == 1))[0]

        if len(idx_close) == 0 or len(idx_open) == 0:
            # 해당 object에 close / open 둘 다 없으면 스킵
            continue

        print(f"    ▶ object {o}: close {len(idx_close)}개, open {len(idx_open)}개")

        for i in idx_close:
            for j in idx_open:
                sim = np.dot(emb_norm[i], emb_norm[j])  # cosine similarity
                spatial.append(sim)

    # 2) Different Object, Same Motion
    for m in [0, 1]:  # close / open
        idx = np.where(motion_id == m)[0]
        if len(idx) < 2:
            continue

        print(f"    ▶ motion {m}: total {len(idx)}개")

        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                # 같은 object는 건너뛰기
                if object_id[idx[i]] == object_id[idx[j]]:
                    continue
                sim = np.dot(emb_norm[idx[i]], emb_norm[idx[j]])
                temporal.append(sim)

    print(f"  ▶ collected pairs (similarity): spatial={len(spatial)}, temporal={len(temporal)}")
    return spatial, temporal

def collect_pair_distances(embeddings, labels):
    # 이 함수는 사용하지 않으므로 변경하지 않습니다.
    # random(8) 제거
    mask = labels != 8
    emb = embeddings[mask]
    lab = labels[mask]

    print("  ▶ unique labels (after removing 8):", np.unique(lab))

    # 라벨 구조: 0~3 close, 4~7 open
    object_id = lab % 4                          # 0,1,2,3
    motion_id = (lab >= 4).astype(np.int64)      # 0(close), 1(open)

    spatial = []
    temporal = []

    # 1) Same Object, Different Motion
    for o in range(4):
        idx_close = np.where((object_id == o) & (motion_id == 0))[0]
        idx_open  = np.where((object_id == o) & (motion_id == 1))[0]

        if len(idx_close) == 0 or len(idx_open) == 0:
            # 해당 object에 close / open 둘 다 없으면 스킵
            continue
        print(f"    ▶ object {o}: close {len(idx_close)}개, open {len(idx_open)}개")
        for i in idx_close:
            for j in idx_open:
                spatial.append(np.linalg.norm(emb[i] - emb[j]))

    # 2) Different Object, Same Motion
    for m in [0, 1]:  # close / open
        idx = np.where(motion_id == m)[0]
        if len(idx) < 2:
            continue
        print(f"    ▶ motion {m}: total {len(idx)}개")
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                if object_id[idx[i]] == object_id[idx[j]]:
                    continue
                temporal.append(np.linalg.norm(emb[idx[i]] - emb[idx[j]]))

    print(f"  ▶ collected pairs: spatial={len(spatial)}, temporal={len(temporal)}")
    return spatial, temporal

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".npz"):
        continue
    
    try:
        epoch_name = fname.replace(".npz", "")
        print(f"\n📂 Processing {epoch_name}")

        data = np.load(os.path.join(input_dir, fname))
        z_video = data["z_video"]
        labels  = data["labels"]

        spatial_sims, temporal_sims = collect_pair_similarities(z_video, labels)

        spatial_sims = np.array(spatial_sims)
        temporal_sims = np.array(temporal_sims)

        print(f"[{epoch_name}] Spatial  - mean {spatial_sims.mean():.4f}, "
              f"std {spatial_sims.std():.4f}, "
              f"min {spatial_sims.min():.4f}, max {spatial_sims.max():.4f}")
        print(f"[{epoch_name}] Temporal - mean {temporal_sims.mean():.4f}, "
              f"std {temporal_sims.std():.4f}, "
              f"min {temporal_sims.min():.4f}, max {temporal_sims.max():.4f}")

        # 쌍이 너무 적으면 그냥 그림 스킵
        if len(spatial_sims) < 5 or len(temporal_sims) < 5:
            print("  ⚠️ too few pairs, skip plotting.")
            continue

        # ==========================================================
        # ★★★ 밀도 플롯 (KDE Plot)으로 교체한 부분 ★★★
        # ==========================================================
        plt.figure(figsize=(8, 6))
        
        # Spatial Pair (Same Obj, Diff Motion) 밀도 플롯
        sns.kdeplot(
            spatial_sims,
            label="Spatial Pair (Same Obj, Diff Motion)",
            color="tab:blue",
            fill=True,
            alpha=0.4,
            linewidth=2
        )
        
        # Temporal Pair (Diff Obj, Same Motion) 밀도 플롯
        sns.kdeplot(
            temporal_sims,
            label="Temporal Pair (Diff Obj, Same Motion)",
            color="tab:orange",
            fill=True,
            alpha=0.4,
            linewidth=2
        )

        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.title(f"Pairwise Similarity Density – {epoch_name}")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # 파일 이름에 'kdeplot' 반영
        out_path = os.path.join(output_dir, f"{epoch_name}_kdeplot_cosine.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"  ✔ Saved: {out_path}")

    except Exception as e:
        print(f"  ❌ Error processing {fname}: {e}")