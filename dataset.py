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
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import random


#####################################################################


class SensorTransform:
    def __init__(self, target_len, filter_btype='low', filter_order=4, cutoff_freq=0.1, 
                interpolation_mode='linear', mean=None, std=None):
        """
        Args:
            target_len (int): The target length of the sequence.
            filter_btype (str): The type of filter ('low', 'high', 'band').
            filter_order (int): The order of the filter.
            cutoff_freq (float or list): The cutoff frequency for the filter.
            interpolation_mode (str): The interpolation mode for resizing.
            mean (np.array, optional): Pre-computed mean for normalization.
            std (np.array, optional): Pre-computed std for normalization.
        """
        self.target_len = target_len
        self.interpolation_mode = interpolation_mode
        self.mean = mean
        self.std = std
        
        # Define Butterworth filter coefficients
        self.b, self.a = butter(filter_order, cutoff_freq, btype=filter_btype, analog=False)

    def fit(self, data_list):
        """
        Calculates mean and std from a list of numpy arrays (training data).
        Args:
            data_list (list): A list of sensor data arrays, each with shape (C, T).
        """
        # Concatenate all data along the time axis
        all_data = np.concatenate(data_list, axis=1)
        # Calculate mean and std channel-wise
        self.mean = np.mean(all_data, axis=1)
        self.std = np.std(all_data, axis=1)
        print("Mean and Std calculated and stored.")

    def _apply_filter(self, data):
        return filtfilt(self.b, self.a, data, axis=1)

    def _apply_normalization(self, data):
        if self.mean is not None and self.std is not None:
            mean = self.mean[:, np.newaxis]
            std = self.std[:, np.newaxis]
            return (data - mean) / (std + 1e-8)
        return data

    def _resize_to_target_len(self, data, device):
        data_tensor = torch.from_numpy(data.copy()).float().to(device)
        data_tensor = data_tensor.unsqueeze(0)
        
        interpolated_tensor = F.interpolate(
            data_tensor, 
            size=self.target_len, 
            mode=self.interpolation_mode, 
            align_corners=False if self.interpolation_mode != 'linear' else None # linear 모드는 align_corners 지원
        )
        return interpolated_tensor.squeeze(0)

    def __call__(self, sensor_data, device='cpu'):
        """
        Args:
            sensor_data (np.array): Input sensor data with shape (C, T).
            device (str): The device to move the final tensor to ('cpu' or 'cuda').
        Returns:
            torch.Tensor: Processed sensor data.
        """
        data_copy = sensor_data.copy()
        
        # 1. Apply normalization
        normalized_data = self._apply_normalization(data_copy)
        
        # 2. Apply filtering
        filtered_data = self._apply_filter(normalized_data)
        
        # 3. Resize the signal
        processed_data = self._resize_to_target_len(filtered_data, device)
        
        return processed_data


#####################################################################


class ClipConsistentTransforms:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        
        # 1. 클립 전체에 대한 랜덤 파라미터 1회 생성
        apply_flip = random.random() < 0.5

        jitter_params = T.ColorJitter.get_params(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4),
            saturation=(0.6, 1.4), hue=(-0.1, 0.1)
        )

        sigma = random.uniform(0.1, 2.0)

        # 2. 모든 프레임에 동일한 파라미터로 변환 적용 (루프)
        tensor_frames = []
        for frame in clip:
            frame = T.Resize(self.size, antialias=True)(frame)

            if apply_flip:
                frame = TF.hflip(frame)
            
            # --- 여기가 수정된 핵심 부분입니다 ---
            # 파라미터를 명확하게 unpacking
            fn_indices, brightness_factor, contrast_factor, saturation_factor, hue_factor = jitter_params

            # 랜덤하게 결정된 함수 순서(fn_indices)대로 순회
            for fn_id in fn_indices:
                if fn_id == 0 and brightness_factor is not None:
                    frame = TF.adjust_brightness(frame, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    frame = TF.adjust_contrast(frame, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    frame = TF.adjust_saturation(frame, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    frame = TF.adjust_hue(frame, hue_factor)
            # --- 수정 끝 ---
            
            frame = TF.gaussian_blur(frame, kernel_size=[5, 5], sigma=sigma)

            tensor_frames.append(T.ToTensor()(frame))

        # 3. 텐서 기반 증강 및 정규화
        # 비디오 데이터는 (C, T, H, W) 또는 (T, C, H, W) 형태가 일반적입니다.
        # torch.stack의 dim 파라미터를 데이터 형태에 맞게 조정하세요.
        # 예: (T, C, H, W)를 원할 경우 dim=0
        clip_tensor = torch.stack(tensor_frames, dim=0)

        if random.random() < 0.5:
            erase_params = T.RandomErasing.get_params(
                clip_tensor, scale=(0.1, 0.2), ratio=(0.3, 3.3), value=[0]
            )
            i, j, h, w, v = erase_params
            clip_tensor = TF.erase(clip_tensor, i, j, h, w, v, inplace=False)

        # Normalize
        clip_tensor = TF.normalize(clip_tensor, mean=self.mean, std=self.std)

        return clip_tensor


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

        # 3. 클립 전체에 대해 한 번에 전처리 적용
        if self.transform:
            # transform이 이제 클립 전체를 받아 최종 텐서를 반환
            frames_tensor = self.transform(frames)
        else:
            # transform이 없는 경우, 기본 ToTensor와 stack만 수행
            frames_tensor = torch.stack([T.ToTensor()(frame) for frame in frames])


        ######### 센서 전처리 #########
        
        # IMU CSV 로드
        df = pd.read_csv(sensor_path)

        selected_indices = np.r_[134:231]

        # .iloc를 사용하여 해당 위치의 컬럼들을 선택합니다.
        selected_df = df.iloc[:, selected_indices].copy() # SettingWithCopyWarning 방지를 위해 .copy()

        # 선택된 데이터프레임에 대해 결측치 처리 시작
        for col in selected_df.columns:
            if selected_df[col].isnull().sum() / len(selected_df) > 0.5:
                selected_df[col].fillna(0, inplace=True)

        selected_df.interpolate(method='linear', limit_direction='both', inplace=True)

        raw = selected_df.values

        sensor_data = raw.T # (C, T)

        # 센서 데이터 전처리 적용
        if self.sensor_transform:
            sensor_data = self.sensor_transform(sensor_data)

        return frames_tensor, sensor_data, label