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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


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

            data=(data-mean*data/(data+1e-8) )/(std+1e-8) # Same with below masking 0 code.

            # mask = np.abs(data) >= 1e-8
            # data[mask]=(data[mask]-self.mean)/self.std
            # 1. 0이 아닌 값들의 위치를 2D 마스크로 찾음

            # mask = np.abs(data) >= 1e-8 

            # # 2. 전체 데이터에 대해 표준화 계산 (NumPy 브로드캐스팅 활용)
            # #    (data(10, 11) - mean(11,)) / std(11,) -> 각 행에 mean/std가 적용됨
            # standardized_data = (data - mean) / (std+1e-8)

            # # 3. 마스크를 사용해서 0이 아니었던 위치의 값들만 표준화된 값으로 업데이트
            # #    data[mask] = standardized_data[mask]와 동일
            # np.copyto(data, standardized_data, where=mask)

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
    def __init__(self, size, mean, std, training=True):
        self.size = size
        self.mean = mean
        self.std = std
        self.training = training

    def __call__(self, clip):
        # Training일 때만 랜덤 파라미터 생성
        if self.training:
            jitter_params = T.ColorJitter.get_params(
                brightness=(0.6, 1.4), contrast=(0.6, 1.4),
                saturation=(0.6, 1.4), hue=(-0.1, 0.1)
            )
            sigma = random.uniform(0.1, 2.0)

        tensor_frames = []
        for frame in clip:
            frame = T.Resize(self.size, antialias=True)(frame)

            # Training일 때만 augmentation 적용
            if self.training:
                fn_indices, brightness_factor, contrast_factor, saturation_factor, hue_factor = jitter_params

                for fn_id in fn_indices:
                    if fn_id == 0 and brightness_factor is not None:
                        frame = TF.adjust_brightness(frame, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        frame = TF.adjust_contrast(frame, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        frame = TF.adjust_saturation(frame, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        frame = TF.adjust_hue(frame, hue_factor)

                frame = TF.gaussian_blur(frame, kernel_size=[5, 5], sigma=sigma)

            tensor_frames.append(T.ToTensor()(frame))
        clip_tensor = torch.stack(tensor_frames, dim=0)
        return clip_tensor


#####################################################################


class VideoSensorDataset(Dataset):
    def __init__(self, json_path: str, data_root: str, num_frames: int, transform, sensor_transform, threshold_epoch, start_index, end_index, cache_dir, use_flow=False, use_cache=False):
        super().__init__()
        
        self.data_root = data_root
        self.num_frames = num_frames
        self.transform = transform
        self.sensor_transform = sensor_transform
        self.current_epoch = 0
        self.threshold_epoch = threshold_epoch
        self.start_index = start_index
        self.end_index = end_index
        self.samples = []
        self.cache_dir = cache_dir
        self.use_cache = use_cache  # val/test용 캐싱 옵션

        # 1. JSON 파일을 읽어 (비디오 전체 경로, 레이블) 리스트 생성
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for item in json_data['data']:

            # JSON에 있는 상대 경로와 데이터 루트 경로를 조합하여 전체 경로 생성
            relative_path_video = item['frame_path']
            relative_path_sensor = item['sensor_path']
            
            video_path = os.path.join(self.data_root, relative_path_video)
            sensor_path = os.path.join(self.data_root, relative_path_sensor)
            
            label = item['label']
            item_id = item['video_id']
            if 'optical_flow_dir' in item:
                relative_path_optical_flow = item['optical_flow_dir']
                optical_flow_path = os.path.join(self.data_root, relative_path_optical_flow)
                print("optical flow path", optical_flow_path)
            else:
                optical_flow_path = None
                # print(self.data_root, item)
            
            self.samples.append((video_path, sensor_path, label, item_id, optical_flow_path))

        self.use_flow = use_flow

    def __len__(self):
        return len(self.samples)

    def _get_cache_path(self, video_path):
        """캐시 파일 경로 생성"""
        parts = video_path.split(os.sep)
        if len(parts) >= 2:
            last_two_parts = '/'.join(parts[-2:])
        else:
            last_two_parts = os.path.basename(video_path)
        cache_path = os.path.join(self.cache_dir, "videos", last_two_parts)
        return cache_path.rsplit('.', 1)[0] + '.pt'

    def _load_from_cache(self, cache_path):
        """캐시에서 텐서 로드"""
        if os.path.exists(cache_path):
            try:
                print("load cache", cache_path)
                return torch.load(cache_path, weights_only=False)
            except Exception as e:
                print(f"Cache load failed: {cache_path}, {e}")
        return None

    def _save_to_cache(self, cache_path, tensor):
        """텐서를 캐시에 저장 (atomic write)"""
        import fcntl
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

    def _load_video_frames(self, video_path):
        """비디오에서 프레임 로드"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at the constructed path: {video_path}")
            raise IOError(f"Cannot open video file, it may be corrupted or in an unsupported format: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 1:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            frame_indices = np.zeros(self.num_frames, dtype=int)
        
        frames = []
        last_successful_frame = None

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                last_successful_frame = frame_pil.copy()
                frames.append(frame_pil)
            else:
                if last_successful_frame is not None:
                    frames.append(last_successful_frame.copy())
                else:
                    frames.append(Image.new('RGB', (224, 224)))
        
        cap.release()
        return frames

    def __getitem__(self, idx: int):
        video_path, sensor_path, label, item_id, flow_path = self.samples[idx]
        
        ######### 비디오 전처리 #########       
        if self.current_epoch <= self.threshold_epoch:
            dummy_clip = [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]
            frames_tensor = self.transform(dummy_clip)
            print("fake clip used", self.current_epoch)

        else:
            # 캐싱 사용 시 (val/test)
            if self.use_cache:
                cache_path = self._get_cache_path(video_path)
                frames_tensor = self._load_from_cache(cache_path)
                
                if frames_tensor is None:
                    # 캐시 miss: 비디오 로드 → transform → 저장
                    frames = self._load_video_frames(video_path)
                    if self.transform:
                        frames_tensor = self.transform(frames)
                    else:
                        frames_tensor = torch.stack([T.ToTensor()(frame) for frame in frames])
                    self._save_to_cache(cache_path, frames_tensor)
            else:
                # 캐싱 미사용 (train): 매번 로드 + 랜덤 augmentation
                frames = self._load_video_frames(video_path)
                if self.transform:
                    frames_tensor = self.transform(frames)
                else:
                    frames_tensor = torch.stack([T.ToTensor()(frame) for frame in frames])


        ######### 센서 전처리 #########
        
        # IMU CSV 로드
        df = pd.read_csv(sensor_path)

        selected_indices = np.r_[self.start_index:self.end_index+1]

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

        # Optical flow 전처리
        if self.use_flow and flow_path is not None:
            try:
                flow_path = os.path.join(flow_path, "flow.npy")
                flow = np.load(flow_path) 
                flow = torch.from_numpy(flow).float()  # [T, 2, H, W]
                # 값 정규화
                T, C, H, W = flow.shape
                if H > 224 or W > 224:
                    # bilinear resize (flow는 벡터이므로 interpolation 모드 주의)
                    flow = F.interpolate(
                        flow, size=(224, 224),
                        mode="bilinear", align_corners=False
                    )
                    print("flow is resized")
            except: 
                print(f"File error: {flow_path}")
                # 파일이 없으면 0으로 채워진 빈 텐서 (크기 [1, 2, H, W] 등)를 생성하거나 
                # 아예 에러를 발생시켜 해당 샘플을 제외하는 것이 더 좋습니다.
                flow = torch.empty(0)
                
        else:
            flow = {}
            print("no flow")
        return frames_tensor, sensor_data, label, [idx, item_id], flow
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch