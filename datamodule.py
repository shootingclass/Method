import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

# --- 사용자 정의 모듈 임포트 ---
from dataset import VideoSensorDataset, SensorTransform, ClipConsistentTransforms
from method_utils import (
    calculate_sensor_stats, save_stats, load_stats
)


####################################################################


class MethodDataModule(pl.LightningDataModule):
    def __init__(self, args, stage='pretrain'):
        super().__init__()

        self.set_dataset_params(args, stage)
        self.num_frames = args.num_frames
        self.threshold_epoch = args.threshold_epoch
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # 프레임 전처리(Transform) 정의
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        
        self.train_transform = ClipConsistentTransforms(
            size=(224, 224),
            mean=clip_mean,
            std=clip_std
        )
    
    def set_dataset_params(self, args, stage):
        if args.dataset_name == "Opportunity++":
            self.data_root = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/"
            self.json_path = f"/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/actionOnlyObject"
            self.stats_file_path = "/mnt/hdd4tb/junho/Opportunity++/sensor_stats/sensor_stats_37.npy"
            self.start_index = 194
            self.end_index = 230
            self.cache_dir = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/caches"
        elif args.dataset_name == "HWU-USP":
            self.data_root = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/"
            self.json_path = f"/mnt/hdd4tb/junho/HWU-USP_v2/splits_with_trashes"
            self.stats_file_path = "/mnt/hdd4tb/junho/HWU-USP_v2/sensor_stats_11.npy"
            self.start_index = 1
            self.end_index = 11
            self.cache_dir = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/caches"
        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")
        
        # --- stage에 따른 JSON 경로 분기 설정 ---
        if stage == 'pretrain':
            print("INFO: DataModule configured for PRE-TRAINING stage.")
            self.json_train_path = os.path.join(self.json_path, "pretrain.json")
            # Pre-training 시 val/test가 필요 없다면 None으로 설정하거나 train과 동일하게 설정
            self.json_val_path = os.path.join(self.json_path, "pretrain.json") if args.model_name == "method" else None
            # Evaluate 용 data를 pretrain data와 동일하게 설정 (leak 방지)
            self.json_test_path = None
        
        elif stage == 'linear_probe':
            print("INFO: DataModule configured for LINEAR PROBING stage.")
            self.json_train_path = os.path.join(self.json_path, "linear_train.json")
            self.json_val_path = os.path.join(self.json_path, "linear_val.json")
            self.json_test_path = os.path.join(self.json_path, "linear_test.json")
        
        else:
            raise ValueError(f"Invalid stage: {stage}. Choose 'pretrain' or 'linear_probe'.")

    # 이 메서드는 단일 프로세스에서만 실행됩니다.
    # 파일 다운로드나 데이터 전처리 등 한 번만 수행해야 할 작업을 여기에 둡니다.
    def prepare_data(self):
        if not os.path.exists(self.stats_file_path):
            print(f"Statistics file not found. Calculating for the first time...")

            temp_dataset = VideoSensorDataset(
                json_path=self.json_train_path,
                data_root=self.data_root,
                num_frames=self.num_frames,
                transform=self.train_transform,
                sensor_transform=None,
                threshold_epoch=self.threshold_epoch,
                start_index=self.start_index,
                end_index=self.end_index,
                cache_dir=self.cache_dir
            )
            stats = calculate_sensor_stats(temp_dataset)
            save_stats(stats, self.stats_file_path)

    # 이 메서드는 모든 GPU에서 각각 실행됩니다.
    # 데이터셋을 여기서 정의합니다.
    def setup(self, stage=None):
        stats = load_stats(self.stats_file_path)
        sensor_preprocessor = SensorTransform(target_len=128, mean=stats['mean'], std=stats['std'])

        if stage == 'fit' or stage is None:
            self.train_dataset = VideoSensorDataset(
                json_path=self.json_train_path,
                data_root=self.data_root,
                num_frames=self.num_frames,
                transform=self.train_transform,
                sensor_transform=sensor_preprocessor,
                threshold_epoch=self.threshold_epoch,
                start_index=self.start_index,
                end_index=self.end_index,
                cache_dir=self.cache_dir
            )
            print(f"Train dataset size: {len(self.train_dataset)}")
            if self.json_val_path: # val 경로가 있을 때만 생성
                self.val_dataset = VideoSensorDataset(
                    json_path=self.json_val_path,
                    data_root=self.data_root,
                    num_frames=self.num_frames,
                    transform=self.train_transform,
                    sensor_transform=sensor_preprocessor,
                    threshold_epoch=self.threshold_epoch,
                    start_index=self.start_index,
                    end_index=self.end_index,
                    cache_dir=self.cache_dir
                )
        
        if stage == 'test' or stage is None:
            if self.json_test_path: # test 경로가 있을 때만 생성
                self.test_dataset = VideoSensorDataset(
                    json_path=self.json_test_path,
                    data_root=self.data_root,
                    num_frames=self.num_frames,
                    transform=self.train_transform,
                    sensor_transform=sensor_preprocessor,
                    threshold_epoch=self.threshold_epoch,
                    start_index=self.start_index,
                    end_index=self.end_index,
                    cache_dir=self.cache_dir
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # VideoSensorDataset의 set_epoch를 호출하기 위한 콜백
    def on_before_train_epoch(self, epoch):
        self.train_dataset.set_epoch(epoch)