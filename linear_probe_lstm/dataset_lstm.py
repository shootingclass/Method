import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SENSOR_COLUMNS = [
    'Ktch_B1_Drawer','Ktch_B2_Cupboard','Ktch_B3_Cupboard','Ktch_B4_Cupboard',
    'Ktch_Motion_1','Ktch_Motion_2','Ktch_T1_Cupboard','Ktch_T2_Cupboard',
    'Ktch_T3_Cupboard','Ktch_T4_Cupboard','TP_L_Power'
]

class MethodDatasetLSTM(Dataset):
    """
    JSON (sequence) 단위로 로드.
    각 item은 하나의 시퀀스:
        - 'windows': 여러 2초 윈도우의 센서 csv 경로들(상대경로)
        - 'class_name'은 시퀀스 내 모든 윈도우에서 동일하다고 가정(아니면 에러)
    반환:
        dict {
            'sensor_seq': FloatTensor [T, C, L]  (C=11, L≈100)
            'length':     int (T)
            'label':      LongTensor ()
        }
    collate_fn에서 [B, T, C, L] 패딩으로 묶어줌.
    """

    def __init__(self, json_path, processed_root, class_to_idx):
        super().__init__()
        self.json_path = json_path
        self.processed_root = processed_root
        self.class_to_idx = class_to_idx

        with open(json_path, 'r') as f:
            obj = json.load(f)

        self.items = []
        for seq in obj["data"]:
            seq_id = seq.get("sequence_id", "")
            windows = seq["windows"]
            # class_name 일관성 검사
            names = {w["class_name"] for w in windows}
            if len(names) != 1:
                raise ValueError(f"class_name mismatch in sequence {seq_id}: {names}")
            class_name = list(names)[0]
            label = self.class_to_idx[class_name]

            # 센서 파일 절대경로 생성
            sensor_paths = [os.path.join(self.processed_root, w["sensor_path"]) for w in windows]
            self.items.append({
                "sequence_id": seq_id,
                "sensor_paths": sensor_paths,
                "label": label
            })

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load_sensor_csv(path):
        # CSV: timestamp + 11센서 → timestamp 제거, shape [L, 11] → [11, L]
        df = pd.read_csv(path)
        if df.shape[1] < 12:
            raise ValueError(f"Unexpected CSV columns in {path}: {df.columns}")
        arr = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # drop timestamp
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # (L, C) -> (C, L)
        return torch.from_numpy(arr.T)  # [C, L]

    def __getitem__(self, idx):
        item = self.items[idx]
        seq_tensors = []
        for sp in item["sensor_paths"]:
            if not os.path.exists(sp):
                # 비어있는 윈도우일 수 있음 → 2초(50Hz) x 11채널 zeros로 대체
                # L=100 가정 (필요시 파일명에서 길이 추정 가능하지만 표준 2초 윈도우로 둠)
                zeros = torch.zeros(len(SENSOR_COLUMNS), 100, dtype=torch.float32)
                seq_tensors.append(zeros)
            else:
                seq_tensors.append(self._load_sensor_csv(sp))

        # 길이 보정(혹시 100이 아닐 수도 있음 → 잘라내거나 패딩)
        fixed_seq = []
        for x in seq_tensors:
            if x.shape[1] < 100:
                pad = torch.zeros(x.shape[0], 100 - x.shape[1])
                x = torch.cat([x, pad], dim=1)
            elif x.shape[1] > 100:
                x = x[:, :100]
            fixed_seq.append(x)

        seq_tensor = torch.stack(fixed_seq, dim=0)  # [T, C, L]
        return {
            "sensor_seq": seq_tensor.float(),
            "length": len(fixed_seq),
            "label": torch.tensor(item["label"]).long(),
        }

    @staticmethod
    def collate_fn(batch):
        """
        batch: list of dicts
        pad T to max_T with zeros on (C,L).
        return:
            sensor_seq: [B, T_max, C, L]
            lengths:    [B]
            label:      [B]
        """
        lengths = [b["length"] for b in batch]
        max_T = max(lengths)
        C = batch[0]["sensor_seq"].shape[1]
        L = batch[0]["sensor_seq"].shape[2]

        B = len(batch)
        out = torch.zeros(B, max_T, C, L, dtype=torch.float32)
        labels = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            t = b["sensor_seq"].shape[0]
            out[i, :t] = b["sensor_seq"]
            labels[i] = b["label"]

        return {
            "sensor_seq": out,      # [B, T, C, L]
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "label": labels
        }
