import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from scipy.signal import butter, filtfilt
import torch.nn.functional as F


#####################################################################


class SensorTransform:
    def __init__(self, target_len, mean=None, std=None, filter_order=4, cutoff_freq=0.1):
        self.target_len = target_len
        self.mean = mean
        self.std = std
        self.filter_order = filter_order
        self.cutoff_freq = cutoff_freq
        
        # Define Butterworth filter coefficients
        self.b, self.a = butter(self.filter_order, self.cutoff_freq, btype='low', analog=False)

    def _apply_filter(self, data):
        # Apply filter along the time axis (axis=1) for each channel
        # data shape: (C, T)
        return filtfilt(self.b, self.a, data, axis=1)

    def _apply_normalization(self, data):
        # Apply channel-wise normalization
        if self.mean is not None and self.std is not None:
            # Ensure mean and std are correctly shaped for broadcasting
            mean = self.mean[:, np.newaxis]
            std = self.std[:, np.newaxis]
            return (data - mean) / (std + 1e-8) # Add epsilon for stability
        return data

    # 보간을 위한 새로운 메서드 (NumPy 배열을 받아 Tensor를 반환)
    def _apply_interpolation(self, data):
        
        # F.interpolate를 위해 NumPy 배열을 Tensor로 변환
        data_tensor = torch.from_numpy(data).float()
        
        # F.interpolate는 [N, C, L] 형태의 3D 텐서를 기대하므로 차원 추가
        data_tensor = data_tensor.unsqueeze(0)  # [1, C, T]
        
        # 선형 보간 적용
        interpolated_tensor = F.interpolate(
            data_tensor, 
            size=self.target_len, 
            mode='linear', 
            align_corners=False
        )
        
        # 추가했던 배치 차원 제거 후 반환
        return interpolated_tensor.squeeze(0) # [C, target_len]

    def __call__(self, sensor_data):
        """
        Args:
            sensor_data (np.array): Input sensor data with shape (C, T).
        Returns:
            np.array: Processed sensor data.
        """
        # 1. Apply filtering
        # Use copy to avoid in-place modification issues and ensure data integrity
        filtered_data = self._apply_filter(sensor_data.copy())
        
        # 2. Apply normalization
        normalized_data = self._apply_normalization(filtered_data)
        
        processed_data = self._apply_interpolation(normalized_data)
        
        return processed_data


#####################################################################


class VideoSensorDataset(Dataset):
    def __init__(self, json_path: str, data_root: str, num_frames: int, transform, sensor_transform):
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.transform = transform
        self.sensor_transform = sensor_transform
        
        self.samples = []

        # 1. JSON 파일을 읽어 (비디오 전체 경로, 레이블) 리스트 생성
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for item in json_data['data']:
            # JSON에 있는 상대 경로와 데이터 루트 경로를 조합하여 전체 경로 생성
            relative_path_video = item['frame_path']
            relative_path_sensor = item['imu_path']
            
            video_path = os.path.join(self.data_root, relative_path_video)
            sensor_path = os.path.join(self.data_root, relative_path_sensor)
            
            label = item['label']
            
            self.samples.append((video_path, sensor_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, sensor_path, label = self.samples[idx]
        
        ######### 비디오 전처리 #########       
        # 1. OpenCV를 사용하여 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at the constructed path: {video_path}")
            raise IOError(f"Cannot open video file, it may be corrupted or in an unsupported format: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 2. 프레임 인덱스 샘플링 (균등 샘플링)
        if total_frames > 1:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # 프레임이 없거나 하나뿐인 비디오 처리
            frame_indices = np.zeros(self.num_frames, dtype=int)
        
        frames = []
        successful_reads = 0
        last_successful_frame = None

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                successful_reads += 1
                # OpenCV(BGR) -> RGB -> PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                last_successful_frame = frame_pil.copy()
                frames.append(frame_pil)
                
            else:
                # 프레임 읽기 실패 시, 마지막으로 성공한 프레임 또는 검은 이미지 사용
                if last_successful_frame is not None:
                    frames.append(last_successful_frame.copy())
                else:
                    frames.append(Image.new('RGB', (224, 224)))

        cap.release()

        # 3. 모든 프레임에 대해 한 번에 전처리 적용
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # 4. 프레임 리스트를 하나의 텐서로 통합
        frames_tensor = torch.stack(frames)


        ######### 센서 전처리 #########
        # IMU CSV 로드
        df = pd.read_csv(sensor_path)
        
        selected_indices = np.r_[134:194, 207:231]

        # .iloc를 사용하여 해당 위치의 컬럼들을 선택하고 .values로 NumPy 배열을 가져옵니다.
        raw = df.iloc[:, selected_indices].values
            
        sensor_data = raw.T # (C, T)

        # 센서 데이터 전처리 적용
        if self.sensor_transform:
            sensor_data = self.sensor_transform(sensor_data)

        return frames_tensor, sensor_data, label