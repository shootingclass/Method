import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# ✅ pretraining에서 쓰던 SensorTransform 불러오기 (relative/absolute fallback)
try:
    from .dataset import SensorTransform
except ImportError:
    from dataset import SensorTransform


# ---------------------------------------------------------------------
# 1️⃣ 센서 컬럼 설정
# ---------------------------------------------------------------------
SENSOR_COLUMNS = [
    'timestamp',
    'Ktch_B1_Drawer', 'Ktch_B2_Cupboard', 'Ktch_B3_Cupboard', 'Ktch_B4_Cupboard',
    'Ktch_Motion_1', 'Ktch_Motion_2', 'Ktch_T1_Cupboard', 'Ktch_T2_Cupboard',
    'Ktch_T3_Cupboard', 'Ktch_T4_Cupboard', 'TP_L_Power'
]

SELECTED_SENSOR_NAMES = [
    'Ktch_B4_Cupboard',
    'Ktch_Motion_1',
    'Ktch_Motion_2',
    'Ktch_T1_Cupboard',
    'Ktch_T2_Cupboard',
    'Ktch_T3_Cupboard',
]


def _pick_column_indices(df_cols: List[str], wanted: List[str]) -> List[int]:
    idxs = []
    for name in wanted:
        if name not in df_cols:
            raise ValueError(f"CSV columns missing required sensor: {name}. Found={df_cols}")
        idxs.append(df_cols.index(name))
    return idxs


# ---------------------------------------------------------------------
# 2️⃣ 데이터 클래스 정의
# ---------------------------------------------------------------------
@dataclass
class SeqItem:
    sequence_id: str
    class_name: str
    class_idx: int
    window_paths: List[str]


class SequenceDataset(Dataset):
    """
    Linear Probe LSTM용 Dataset.
    JSON 구조:
    {
        "data": [
            {
                "sequence_id": "tidy_s14",
                "windows": [
                    {"sensor_path": "trim_2s_sensor/tidy/...csv", "class_name": "tidy", ...},
                    ...
                ]
            }
        ]
    }
    """
    def __init__(self, json_path: str, data_root: str, class_to_idx: Dict[str, int],
                 dtype: torch.dtype = torch.float32, sensor_transform: SensorTransform = None):
        super().__init__()
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.dtype = dtype
        self.sensor_transform = sensor_transform

        with open(json_path, 'r', encoding='utf-8') as f:
            j = json.load(f)

        self.items: List[SeqItem] = []
        print("init! ")
        print("")
        for seq in j["data"]:
            if "sequence_id" not in seq or "windows" not in seq:
                raise ValueError(f"Invalid sequence entry in {json_path}: {seq.keys()}")

            seq_id = seq["sequence_id"]
            windows = seq["windows"]
            if len(windows) == 0:
                continue

            class_names = set([w["class_name"] for w in windows])
            if len(class_names) != 1:
                raise ValueError(f"[{seq_id}] sequence has multiple class_name: {class_names}.")
            class_name = list(class_names)[0]

            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
                print("load ", class_name)
            class_idx = self.class_to_idx[class_name]

            window_csvs = [os.path.join(self.data_root, w["sensor_path"]) for w in windows]
            self.items.append(SeqItem(seq_id, class_name, class_idx, window_csvs))
            print("sequnce id and class name", seq_id, class_name)

    def __len__(self):
        return len(self.items)

    def _load_window_csv(self, csv_path: str) -> np.ndarray:
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        selected_indices = _pick_column_indices(cols, SELECTED_SENSOR_NAMES)
        selected_df = df.iloc[:, selected_indices].fillna(0)
        data = selected_df.values.astype(np.float32).T  # (C, T)
        if data.shape[1] < 100:
            return None
        return data

    def __getitem__(self, idx: int):
        item = self.items[idx]
        tensors: List[torch.Tensor] = []

        for csv_path in item.window_paths:
            arr = self._load_window_csv(csv_path)
            if arr is None:
                continue

            # ✅ 동일한 SensorTransform 파이프라인 사용
            if self.sensor_transform is not None:
                arr = self.sensor_transform(arr)  # torch.Tensor (C, target_len)
                if not isinstance(arr, torch.Tensor):
                    arr = torch.from_numpy(np.asarray(arr))
            else:
                arr = torch.from_numpy(arr)

            tensors.append(arr.to(dtype=self.dtype))

        # ✅ 모든 윈도우가 너무 짧을 경우 dummy 생성
        if len(tensors) == 0:
            print("[WARN] too small window!!!")
            C = len(SELECTED_SENSOR_NAMES)
            target_len = getattr(self.sensor_transform, "target_len", 128)
            tensors.append(torch.zeros((C, target_len), dtype=self.dtype))
    
        seq_windows = torch.stack(tensors, dim=0).contiguous()  # (S, C, T)
         # 🔍 DEBUG SHAPE
        if seq_windows.shape[1] != 6 or seq_windows.shape[2] != 128:
            print(f"[WARN] Irregular shape in seq {item.sequence_id}: {seq_windows.shape}")

        target = torch.tensor(item.class_idx, dtype=torch.long)
        meta = {"sequence_id": item.sequence_id, "length": seq_windows.shape[0]}
        return seq_windows, target, meta


# ---------------------------------------------------------------------
# 3️⃣ Collate function (variable-length 지원)
# ---------------------------------------------------------------------
def collate_variable_length(batch):
    sequences, targets, metas = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    B = len(sequences)
    S_max = int(max(lengths).item())
    C = sequences[0].shape[1]
    T = sequences[0].shape[2]

    padded = torch.zeros((B, S_max, C, T), dtype=sequences[0].dtype)
    for i, s in enumerate(sequences):
        padded[i, :s.shape[0]] = s

    targets = torch.stack(targets, dim=0)
    return padded, lengths, targets, list(metas)


# ---------------------------------------------------------------------
# 4️⃣ Lightning DataModule
# ---------------------------------------------------------------------
class LinearProbeLSTMDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=8, pin_memory=True, sensor_transform=None):
        super().__init__()
        self.data_root = Path("/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window")
        base = self.data_root.parent / "motion_2_almost_priority"
        self.train_json = base / "linear_probe_train.json"
        self.test_json = base / "linear_probe_test.json"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.class_to_idx: Dict[str, int] = {}
        self.sensor_transform = sensor_transform

    def setup(self, stage=None):
        print(f"✅ [HWU-USP] Loading sequence data from: {self.data_root}")
        self.train_set = SequenceDataset(
            self.train_json, self.data_root, self.class_to_idx, sensor_transform=self.sensor_transform
        )
        self.test_set = SequenceDataset(
            self.test_json, self.data_root, self.class_to_idx, sensor_transform=self.sensor_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_variable_length,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_variable_length,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()
