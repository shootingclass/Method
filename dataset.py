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

import torch
import numpy as np
# from scipy.signal import butter, filtfilt # 더 이상 필요 없음
# import torch.nn.functional as F # 더 이상 필요 없음
import torch
import numpy as np

class SensorTransform:
    def __init__(self, target_len, mean=None, std=None):
        """
        데이터 전처리의 모든 단계를 포함하는 최종 버전.
        정규화 -> 패딩/절단 -> NaN 처리 -> Tensor 변환 (float32)
        """
        self.target_len = target_len
        self.mean = mean
        self.std = std

    def _apply_normalization(self, data):
        """채널별 정규화를 적용합니다."""
        if self.mean is not None and self.std is not None:
            mean = self.mean[:, np.newaxis]
            std = self.std[:, np.newaxis]
            return (data - mean) / (std + 1e-8)
        return data

    def _apply_padding_and_truncation(self, data):
        """데이터 길이를 target_len에 맞추기 위해 패딩 또는 절단을 적용합니다."""
        num_channels, current_len = data.shape
        
        if current_len < self.target_len:
            padded_data = np.zeros((num_channels, self.target_len))
            padded_data[:, :current_len] = data
            return padded_data
        elif current_len > self.target_len:
            return data[:, :self.target_len]
        else:
            return data

    def __call__(self, sensor_data):
        """
        모든 전처리 파이프라인을 실행하고 최종 텐서를 반환합니다.
        
        Args:
            sensor_data (np.array): (C, T) 형태의 입력 센서 데이터
        Returns:
            torch.Tensor: (C, target_len) 형태의 최종 텐서 (float32)
        """
        # 1. 정규화 적용
        normalized_data = self._apply_normalization(sensor_data.copy())
        
        # 2. 패딩 또는 절단 적용
        processed_data = self._apply_padding_and_truncation(normalized_data)
        
        # 3. NaN 값 확인 및 0으로 대체 (안전장치)
        if np.isnan(processed_data).any():
            # 이 메시지가 보이면 전처리 과정 어딘가에 문제가 있다는 신호입니다.
            print("Warning: NaN detected after processing. Converting to 0.")
            processed_data = np.nan_to_num(processed_data, nan=0.0)

        # 4. 최종 결과를 PyTorch Tensor로 변환하고 dtype을 float32로 명시
        # 이 단계에서 dtype을 통일하여 'mixed dtype' 에러를 근본적으로 방지합니다.
        final_tensor = torch.from_numpy(processed_data).to(torch.float32)
        
        return final_tensor

# class SensorTransform:
#     def __init__(self, target_len, mean=None, std=None, filter_order=4, cutoff_freq=0.1):
#         self.target_len = target_len
#         self.mean = mean
#         self.std = std
#         self.filter_order = filter_order
#         self.cutoff_freq = cutoff_freq
        
#         # Define Butterworth filter coefficients
#         self.b, self.a = butter(self.filter_order, self.cutoff_freq, btype='low', analog=False)

#     def _apply_filter(self, data):
#         # Apply filter along the time axis (axis=1) for each channel
#         # data shape: (C, T)
#         return filtfilt(self.b, self.a, data, axis=1)

#     def _apply_normalization(self, data):
#         # Apply channel-wise normalization
#         if self.mean is not None and self.std is not None:
#             # Ensure mean and std are correctly shaped for broadcasting
#             mean = self.mean[:, np.newaxis]
#             std = self.std[:, np.newaxis]
#             return (data - mean) / (std + 1e-8) # Add epsilon for stability
#         return data

#     # 보간을 위한 새로운 메서드 (NumPy 배열을 받아 Tensor를 반환)
#     def _apply_interpolation(self, data):
        
#         # F.interpolate를 위해 NumPy 배열을 Tensor로 변환
#         data_tensor = torch.from_numpy(data).float()
        
#         # F.interpolate는 [N, C, L] 형태의 3D 텐서를 기대하므로 차원 추가
#         data_tensor = data_tensor.unsqueeze(0)  # [1, C, T]
        
#         # 선형 보간 적용
#         interpolated_tensor = F.interpolate(
#             data_tensor, 
#             size=self.target_len, 
#             mode='linear', 
#             align_corners=False
#         )
        
#         # 추가했던 배치 차원 제거 후 반환
#         return interpolated_tensor.squeeze(0) # [C, target_len]

#     def __call__(self, sensor_data):
#         """
#         Args:
#             sensor_data (np.array): Input sensor data with shape (C, T).
#         Returns:
#             np.array: Processed sensor data.
#         """
#         # 1. Apply filtering
#         # Use copy to avoid in-place modification issues and ensure data integrity
#         filtered_data = self._apply_filter(sensor_data.copy())
        
#         # 2. Apply normalization
#         normalized_data = self._apply_normalization(filtered_data)
        
#         processed_data = self._apply_interpolation(normalized_data)
        
#         return processed_data


#####################################################################


class VideoSensorDataset(Dataset):
    def __init__(self, json_path: str, data_root: str, num_frames: int, transform, sensor_transform, threshold_epoch=6):
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.transform = transform
        self.sensor_transform = sensor_transform
        self.threshold_epoch = threshold_epoch
        
        self.samples = []
        self.current_epoch = 0

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
        if self.current_epoch < self.threshold_epoch:  # 첫 6 epoch는 가짜 프레임 생성
        # 가짜 프레임 생성
            # 1. transform의 출력 텐서 모양을 파악합니다.
            #    (예: 3채널, 224x224 크기의 이미지)
            dummy_frame = self.transform(Image.new('RGB', (224, 224)))
            C, H, W = dummy_frame.shape

            # 2. 원하는 shape (프레임 수, 채널, 높이, 너비)로 zero 텐서를 만듭니다.
            frames_tensor = torch.zeros(self.num_frames, C, H, W)
        
        else:
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
        # print("sensor_path", sensor_path)
        
        selected_indices = np.r_[134:231]

        # .iloc를 사용하여 해당 위치의 컬럼들을 선택하고 .values로 NumPy 배열을 가져옵니다.
        raw = df.iloc[:, selected_indices].values
            
        sensor_data = raw.T # (C, T)

        # 센서 데이터 전처리 적용
        if self.sensor_transform:
            sensor_data = self.sensor_transform(sensor_data)

        return frames_tensor, sensor_data, label

    def set_epoch(self, epoch):
        self.current_epoch = epoch