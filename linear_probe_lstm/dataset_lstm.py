import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path


# 고정된 센서 컬럼 (CSV 헤더 기준)
SENSOR_COLUMNS = [
    'timestamp',
    'Ktch_B1_Drawer', 'Ktch_B2_Cupboard', 'Ktch_B3_Cupboard', 'Ktch_B4_Cupboard',
    'Ktch_Motion_1', 'Ktch_Motion_2', 'Ktch_T1_Cupboard', 'Ktch_T2_Cupboard',
    'Ktch_T3_Cupboard', 'Ktch_T4_Cupboard', 'TP_L_Power'
]

# 사용할 센서만 선택 (TP_L_Power 제외)
SELECTED_SENSOR_NAMES = [
    'Ktch_B4_Cupboard',
    'Ktch_Motion_1',
    'Ktch_Motion_2',
    'Ktch_T1_Cupboard',
    'Ktch_T2_Cupboard',
    'Ktch_T3_Cupboard',
]
# DF 내 컬럼 인덱스를 헤더 이름으로 안전하게 찾는다.
def _pick_column_indices(df_cols: List[str], wanted: List[str]) -> List[int]:
    idxs = []
    for name in wanted:
        if name not in df_cols:
            raise ValueError(f"CSV columns missing required sensor: {name}. Found={df_cols}")
        idxs.append(df_cols.index(name))
    return idxs


@dataclass
class SeqItem:
    sequence_id: str
    class_name: str
    class_idx: int
    window_paths: List[str]  # CSV 경로 리스트


class SequenceDataset(Dataset):
    """
    JSON (sequence 형식) 을 읽어, 한 샘플 = 하나의 시퀀스.
    각 시퀀스는 여러 윈도우 CSV를 가진다. (overlap 허용)
    목표 라벨은 sequence-level 'class_name' (활동 클래스).
    """
    def __init__(self,
                 json_path: str,
                 data_root: str,
                 class_to_idx: Dict[str, int],
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.dtype = dtype

        with open(json_path, 'r', encoding='utf-8') as f:
            j = json.load(f)

        self.items: List[SeqItem] = []
        for seq in j["data"]:
            # 형식 체크
            if "sequence_id" not in seq or "windows" not in seq:
                raise ValueError(f"Invalid sequence entry in {json_path}: {seq.keys()}")

            seq_id = seq["sequence_id"]
            windows = seq["windows"]
            if len(windows) == 0:
                # 빈 시퀀스는 스킵
                continue

            # 같은 시퀀스 내 class_name들이 모두 동일해야 함 (요청사항)
            class_names = set([w["class_name"] for w in windows])
            if len(class_names) != 1:
                raise ValueError(
                    f"[{seq_id}] sequence has multiple class_name: {class_names}. "
                    "All windows in a sequence must share the same class_name."
                )
            class_name = list(class_names)[0]
            if class_name not in self.class_to_idx:
                # 새 클래스를 발견하면 add (train/test를 별개 로드할 때 안전)
                self.class_to_idx[class_name] = len(self.class_to_idx)
            class_idx = self.class_to_idx[class_name]

            window_csvs = [os.path.join(self.data_root, w["sensor_path"]) for w in windows]
            self.items.append(SeqItem(
                sequence_id=seq_id,
                class_name=class_name,
                class_idx=class_idx,
                window_paths=window_csvs
            ))

    def __len__(self) -> int:
        return len(self.items)

    def _load_window_csv(self, csv_path: str) -> np.ndarray:
        """
        CSV 하나를 (C, T) np.float32 로 로드 (선택된 센서만).
        - 결측치는 0으로 채움
        - 길이가 0인 경우 (빈 파일)도 (C, T=100) 형태로 0 패딩 (2초@50Hz) 처리
        - timestamp 열은 무시, SELECTED_SENSOR_NAMES 만 사용
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Sensor CSV not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV {csv_path}: {e}")

        if df is None or df.shape[0] == 0:
            # 완전 빈 파일 → 2초(100스텝) 0 패딩
            C = len(SELECTED_SENSOR_NAMES)
            return np.zeros((C, 100), dtype=np.float32)

        # 헤더 보정: timestamp 포함되어야 함
        cols = list(df.columns)
        if "timestamp" not in cols:
            raise ValueError(f"{csv_path}: 'timestamp' column missing.")

        # 선택 컬럼 확보
        wanted_cols = SELECTED_SENSOR_NAMES
        selected_indices = _pick_column_indices(cols, wanted_cols)
        selected_df = df.iloc[:, selected_indices].copy()

        # 결측치는 0으로
        selected_df = selected_df.fillna(0)

        # (T, C) -> (C, T)
        data = selected_df.values.astype(np.float32).T

        # 길이 0인 경우 방어
        if data.shape[1] == 0:
            data = np.zeros((len(wanted_cols), 100), dtype=np.float32)

        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            seq_windows: [S, C, T] float tensor
            target:      [ ] class index (long)
            meta:        dict{ 'sequence_id': str, 'length': S }
        """
        item = self.items[idx]
        tensors: List[torch.Tensor] = []
        for csv_path in item.window_paths:
            arr = self._load_window_csv(csv_path)  # (C, T)
            tensors.append(torch.from_numpy(arr).to(self.dtype))
        # [S, C, T]
        seq_windows = torch.stack(tensors, dim=0)
        target = torch.tensor(item.class_idx, dtype=torch.long)
        meta = {"sequence_id": item.sequence_id, "length": seq_windows.shape[0]}
        return seq_windows, target, meta


def collate_variable_length(batch):
    """
    배치 내부 시퀀스 길이가 제각각일 때 패딩 + length 반환
    Input: list of (seq_windows[S_i,C,T], target, meta)
    Output:
        padded: [B, S_max, C, T]
        lengths: [B]
        targets: [B]
        metas: list[dict]
    """
    sequences, targets, metas = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    B = len(sequences)
    S_max = int(max(lengths).item())
    C = sequences[0].shape[1]
    T = sequences[0].shape[2]

    padded = torch.zeros((B, S_max, C, T), dtype=sequences[0].dtype)
    for i, s in enumerate(sequences):
        cur_len = s.shape[0]
        padded[i, :cur_len] = s

    targets = torch.stack(targets, dim=0)
    return padded, lengths, targets, list(metas)


class LinearProbeLSTMDatamodule(pl.LightningDataModule):
    """
    HWU-USP용 sequence-level datamodule.
    JSON 경로, data_root 모두 내부에서 자동 설정됨.
    """
    def __init__(self,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 pin_memory: bool = True):
        super().__init__()
        # ✅ 내부에서 자동 설정
        self.data_root = Path("/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window")
        self.train_json = os.path.join(
            self.data_root.parent, "motion_2_almost_priority/linear_probe_train.json"
        )
        self.test_json = os.path.join(
            self.data_root.parent, "motion_2_almost_priority/linear_probe_test.json"
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # 클래스 매핑은 train 로딩 시 자동 생성
        self.class_to_idx: Dict[str, int] = {}

    def setup(self, stage=None):
        print(f"✅ [HWU-USP] Loading sequence data from: {self.data_root}")
        print(f"   Train JSON: {self.train_json}")
        print(f"   Test  JSON: {self.test_json}")

        self.train_set = SequenceDataset(
            json_path=self.train_json,
            data_root=self.data_root,
            class_to_idx=self.class_to_idx
        )
        self.test_set = SequenceDataset(
            json_path=self.test_json,
            data_root=self.data_root,
            class_to_idx=self.class_to_idx
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_variable_length,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_variable_length,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_variable_length,
            persistent_workers=self.num_workers > 0
        )
