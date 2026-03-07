import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import cv2
import fcntl
import torch
import torch.nn.functional as F
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
    video_paths: List[str]  # video 경로 추가
    flow_paths: List[str]   # optical flow 경로 추가


class SequenceDataset(Dataset):
    """
    Linear Probe LSTM용 Dataset.
    JSON 구조:
    {
        "data": [
            {
                "sequence_id": "tidy_s14",
                "windows": [
                    {"sensor_path": "trim_2s_sensor/tidy/...csv", "frame_path": "...", "class_name": "tidy", ...},
                    ...
                ]
            }
        ]
    }
    """
    def __init__(self, json_path: str, data_root: str, class_to_idx: Dict[str, int],
                 dtype: torch.dtype = torch.float32, sensor_transform: SensorTransform = None,
                 include_video: bool = False, num_frames: int = 16, video_transform=None,
                 cache_dir: str = None, use_flow: bool = False, use_cache: bool = False):
        super().__init__()
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.dtype = dtype
        self.sensor_transform = sensor_transform
        self.include_video = include_video
        self.num_frames = num_frames
        self.video_transform = video_transform
        self.use_flow = use_flow
        self.cache_dir = cache_dir or os.path.join(data_root, "caches", "videos")
        self.use_cache = use_cache  # val/test용 캐싱 옵션

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
            video_paths = [os.path.join(self.data_root, w.get("frame_path", "")) for w in windows]
            flow_paths = [os.path.join(self.data_root, w.get("optical_flow_dir", "")) if w.get("optical_flow_dir") else "" for w in windows]
            self.items.append(SeqItem(seq_id, class_name, class_idx, window_csvs, video_paths, flow_paths))
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

    def _get_cache_path(self, video_path: str) -> str:
        """캐시 파일 경로 생성"""
        rel_path = os.path.relpath(video_path, self.data_root)
        cache_path = os.path.join(self.cache_dir, rel_path.rsplit('.', 1)[0] + '.pt')
        return cache_path

    def _load_from_cache(self, cache_path: str) -> torch.Tensor:
        """캐시에서 텐서 로드"""
        if os.path.exists(cache_path):
            try:
                print("load cache", cache_path)
                return torch.load(cache_path, weights_only=False)
            except Exception as e:
                print(f"Cache load failed: {cache_path}, {e}")
        return None

    def _save_to_cache(self, cache_path: str, tensor: torch.Tensor):
        """텐서를 캐시에 저장 (atomic write)"""
        try:
            cache_dir = os.path.dirname(cache_path)
            os.makedirs(cache_dir, exist_ok=True)
            
            lock_path = cache_path + ".lock"
            with open(lock_path, 'w') as f_lock:
                fcntl.flock(f_lock, fcntl.LOCK_EX)
                
                if not os.path.exists(cache_path):
                    temp_path = cache_path + ".tmp"
                    torch.save(tensor, temp_path)
                    os.rename(temp_path, cache_path)
                    
            if os.path.exists(lock_path):
                os.remove(lock_path)
            print("save cache", cache_path)
        except Exception as e:
            print(f"Cache save failed: {cache_path}, {e}")

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """비디오 프레임 로딩 (캐싱 지원)"""
        
        if not os.path.exists(video_path):
            return None
        
        # 캐싱 사용 시 캐시 확인
        if self.use_cache:
            cache_path = self._get_cache_path(video_path)
            cached_tensor = self._load_from_cache(cache_path)
            if cached_tensor is not None:
                return cached_tensor
        
        # 비디오에서 프레임 추출
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        
        if len(frames) < self.num_frames:
            return None
        
        # transform 적용
        if self.video_transform is not None:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            transformed = self.video_transform(pil_frames)
            if isinstance(transformed, torch.Tensor):
                result = transformed
            else:
                result = torch.from_numpy(transformed).permute(0, 3, 1, 2).float()
        else:
            frames_np = np.stack(frames, axis=0)
            result = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        
        # 캐싱 사용 시 저장
        if self.use_cache:
            self._save_to_cache(cache_path, result)
        
        return result

    def _load_flow_frames(self, flow_dir: str) -> torch.Tensor:
        """Optical flow 프레임 로딩 (flow.npy 파일)"""
        if not flow_dir or not os.path.exists(flow_dir):
            return None
        
        # flow.npy 파일 경로
        flow_path = os.path.join(flow_dir, "flow.npy")
        if not os.path.exists(flow_path):
            return None
        
        try:
            flow = np.load(flow_path)  # (T, 2, H, W)
            flow = torch.from_numpy(flow).float()
            
            # shape 확인
            if len(flow.shape) != 4:
                print(f"[WARN] Unexpected flow shape: {flow.shape}")
                return None
            
            T, C, H, W = flow.shape
            
            # 224x224로 resize
            if H != 224 or W != 224:
                flow = F.interpolate(
                    flow, size=(224, 224),
                    mode="bilinear", align_corners=False
                )
            
            # num_frames개로 샘플링
            if T != self.num_frames:
                indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
                flow = flow[indices]  # (num_frames, C, H, W)
            
            return flow  # (T, C, H, W)
        except Exception as e:
            print(f"[WARN] Flow load error: {flow_path}, {e}")
            return None

    def __getitem__(self, idx: int):
        print("getitem", idx)
        item = self.items[idx]
        sensor_tensors: List[torch.Tensor] = []
        video_tensors: List[torch.Tensor] = [] if self.include_video else None
        flow_tensors: List[torch.Tensor] = [] if self.include_video else None

        for i, csv_path in enumerate(item.window_paths):
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

            sensor_tensors.append(arr.to(dtype=self.dtype))
            
            # 비디오 로딩
            if self.include_video and i < len(item.video_paths):
                video_path = item.video_paths[i]
                video_frames = self._load_video_frames(video_path)
                if video_frames is not None:
                    video_tensors.append(video_frames)
                else:
                    # 비디오가 없으면 dummy
                    print("[WARN] video not found: ", video_path)
                    video_tensors.append(torch.zeros((self.num_frames, 3, 224, 224), dtype=self.dtype))
                
                # Flow 로딩
                if self.use_flow and i < len(item.flow_paths) and item.flow_paths[i]:
                    flow_frames = self._load_flow_frames(item.flow_paths[i])
                    if flow_frames is not None:
                        flow_tensors.append(flow_frames)
                        # print("flow is used", item.flow_paths[i])
                    else:
                        # flow가 없으면 None
                        print("[WARN] flow not found: ", item.flow_paths[i])
                        flow_tensors.append(None)
                else:
                    # flow path가 없으면 None
                    # print("[WARN] Don't use flow: ", item.flow_paths[i] if i < len(item.flow_paths) else "N/A")
                    flow_tensors.append(None)

        # ✅ 모든 윈도우가 너무 짧을 경우 dummy 생성
        if len(sensor_tensors) == 0:
            print("[WARN] too small window!!!")
            C = len(SELECTED_SENSOR_NAMES)
            target_len = getattr(self.sensor_transform, "target_len", 128)
            sensor_tensors.append(torch.zeros((C, target_len), dtype=self.dtype))
            if self.include_video:
                video_tensors.append(torch.zeros((self.num_frames, 3, 224, 224), dtype=self.dtype))
                flow_tensors.append(None)  # flow는 None으로
    
        seq_windows = torch.stack(sensor_tensors, dim=0).contiguous()  # (S, C, T)
         # 🔍 DEBUG SHAPE
        if seq_windows.shape[1] != 6 or seq_windows.shape[2] != 128:
            print(f"[WARN] Irregular shape in seq {item.sequence_id}: {seq_windows.shape}")

        target = torch.tensor(item.class_idx, dtype=torch.long)
        
        if self.include_video:
            video_windows = torch.stack(video_tensors, dim=0).contiguous()  # (S, T, C, H, W)
            
            # flow_tensors에 None이 하나라도 있으면 flow_windows는 None
            if any(f is None for f in flow_tensors):
                flow_windows = None
                flow_valid = False
                print(f"[INFO] flow is None for sequence: {item.sequence_id}")
            else:
                flow_windows = torch.stack(flow_tensors, dim=0).contiguous()  # (S, T, C_flow, H, W)
                flow_valid = True
            
            meta = {"sequence_id": item.sequence_id, "length": seq_windows.shape[0], "flow_valid": flow_valid}
            return seq_windows, video_windows, flow_windows, target, meta
        else:
            meta = {"sequence_id": item.sequence_id, "length": seq_windows.shape[0]}
            return seq_windows, target, meta


# ---------------------------------------------------------------------
# 3️⃣ Collate function (variable-length 지원)
# ---------------------------------------------------------------------
def collate_variable_length(batch):
    # batch 형식 확인: video+flow 포함 여부
    # 5개: (sensor, video, flow, target, meta) - video+flow 포함
    # 3개: (sensor, target, meta) - sensor만
    has_video = len(batch[0]) == 5
    
    if has_video:
        sensor_seqs, video_seqs, flow_seqs, targets, metas = zip(*batch)
    else:
        sensor_seqs, targets, metas = zip(*batch)
        video_seqs = None
        flow_seqs = None
    
    lengths = torch.tensor([s.shape[0] for s in sensor_seqs], dtype=torch.long)
    B = len(sensor_seqs)
    S_max = int(max(lengths).item())
    C = sensor_seqs[0].shape[1]
    T_sensor = sensor_seqs[0].shape[2]

    # 센서 패딩
    sensor_padded = torch.zeros((B, S_max, C, T_sensor), dtype=sensor_seqs[0].dtype)
    for i, s in enumerate(sensor_seqs):
        sensor_padded[i, :s.shape[0]] = s

    targets = torch.stack(targets, dim=0)
    
    if has_video:
        # video shape: (S, T_video, C_video, H, W)
        T_video = video_seqs[0].shape[1]
        C_video = video_seqs[0].shape[2]
        H = video_seqs[0].shape[3]
        W = video_seqs[0].shape[4]
        
        video_padded = torch.zeros((B, S_max, T_video, C_video, H, W), dtype=video_seqs[0].dtype)
        for i, v in enumerate(video_seqs):
            video_padded[i, :v.shape[0]] = v
        
        # flow: None이 하나라도 있으면 전체를 None으로
        if any(f is None for f in flow_seqs):
            flow_padded = None
        else:
            # flow shape: (S, T_flow, C_flow, H, W)
            T_flow = flow_seqs[0].shape[1]
            C_flow = flow_seqs[0].shape[2]
            H_flow = flow_seqs[0].shape[3]
            W_flow = flow_seqs[0].shape[4]
            
            flow_padded = torch.zeros((B, S_max, T_flow, C_flow, H_flow, W_flow), dtype=flow_seqs[0].dtype)
            for i, f in enumerate(flow_seqs):
                flow_padded[i, :f.shape[0]] = f
        
        return sensor_padded, video_padded, flow_padded, lengths, targets, list(metas)
    else:
        return sensor_padded, lengths, targets, list(metas)


# ---------------------------------------------------------------------
# 4️⃣ Lightning DataModule
# ---------------------------------------------------------------------
class LinearProbeLSTMDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=8, pin_memory=True, sensor_transform=None):
        super().__init__()
        self.data_root = Path("/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window")
        base = self.data_root / "motion_2_almost_priority_test=18"
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
