import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

# --- 사용자 정의 모듈 임포트 ---
from dataset import VideoSensorDataset, SensorTransform, ClipConsistentTransforms
from utils import (
    calculate_sensor_stats, save_stats, load_stats
)


####################################################################


class MethodDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = self.set_dataset_params(args)
        self.save_hyperparameters(self.args)
        
        # 프레임 전처리(Transform) 정의
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        
        self.train_transform = ClipConsistentTransforms(
            size=(224, 224),
            mean=clip_mean,
            std=clip_std
        )
    
    def set_dataset_params(self, args):
        if args.dataset_name == "Opportunity++":
            args.data_root = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/"
            args.json_train_path = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/actionOnlyObject/pretrain.json"
            # args.stats_file_path = "/mnt/hdd4tb/junho/Opportunity++/sensor_stats/sensor_stats_37.npy"
            args.stats_file_path = "/home/jaemo/channel_stats/sensor_stats_37.npy"
            args.start_index = 194
            args.end_index = 230
        elif args.dataset_name == "HWU-USP":
            args.data_root = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/"
            args.json_train_path = "/mnt/hdd4tb/junho/HWU-USP_v2/pretrain.json"
            # args.stats_file_path = "/mnt/hdd4tb/junho/HWU-USP_v2/sensor_stats_11_activate=0.npy"
            args.stats_file_path = "/home/jaemo/channel_stats/sensor_stats_11.npy"
            args.start_index = 1
            args.end_index = 11
        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")
        return args

    # 이 메서드는 단일 프로세스에서만 실행됩니다.
    # 파일 다운로드나 데이터 전처리 등 한 번만 수행해야 할 작업을 여기에 둡니다.
    def prepare_data(self):
        if not os.path.exists(self.hparams.stats_file_path):
            print(f"Statistics file not found. Calculating for the first time...")

            temp_dataset = VideoSensorDataset(
                json_path=self.hparams.json_train_path,
                data_root=self.hparams.data_root,
                num_frames=self.hparams.num_frames,
                transform=self.train_transform,
                sensor_transform=None,
                threshold_epoch=self.hparams.threshold_epoch,
                start_index=self.args.start_index,
                end_index=self.args.end_index
            )
            stats = calculate_sensor_stats(temp_dataset)
            save_stats(stats, self.hparams.stats_file_path)

    # 이 메서드는 모든 GPU에서 각각 실행됩니다.
    # 데이터셋을 여기서 정의합니다.
    def setup(self, stage=None):
        stats = load_stats(self.hparams.stats_file_path)
        sensor_preprocessor = SensorTransform(target_len=128, mean=stats['mean'], std=stats['std'])

        if stage == 'fit' or stage is None:
            self.train_dataset = VideoSensorDataset(
                json_path=self.hparams.json_train_path,
                data_root=self.hparams.data_root,
                num_frames=self.hparams.num_frames,
                transform=self.train_transform,
                sensor_transform=sensor_preprocessor,
                threshold_epoch=self.hparams.threshold_epoch,
                start_index=self.args.start_index,
                end_index=self.args.end_index
            )
            print(f"Train dataset size: {len(self.train_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # VideoSensorDataset의 set_epoch를 호출하기 위한 콜백
    def on_before_train_epoch(self, epoch):
        self.train_dataset.set_epoch(epoch)