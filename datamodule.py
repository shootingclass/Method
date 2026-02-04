import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

# --- 사용자 정의 모듈 임포트 ---
from dataset import VideoSensorDataset, SensorTransform, ClipConsistentTransforms
from dataset_lstm import SequenceDataset, collate_variable_length
from method_utils import (
    calculate_sensor_stats, save_stats, load_stats
)


####################################################################


class MethodDataModule(pl.LightningDataModule):
    def __init__(self, args, stage='pretrain'):
        super().__init__()

        self.set_dataset_params(args, stage)
        self.num_frames = args.num_frames if hasattr(args, "num_frames") else 20
        if args.model_name=="method":
            self.threshold_epoch = args.threshold_epoch
        else:
            self.threshold_epoch = -1
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.encoder_type = getattr(args, 'encoder_type', 'sensor')  # encoder_type 저장
        self.use_flow = getattr(args, 'use_flow', False)

        # 프레임 전처리(Transform) 정의 
        # CLIP
        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std = [0.26862954, 0.26130258, 0.27577711]

        # Conv 기반 model 전처리
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        
        self.train_transform = ClipConsistentTransforms(
            size=(224, 224),
            mean=mean,
            std=std
        )
        self.stage = stage
    
    def set_dataset_params(self, args, stage):
        if args.dataset_name == "Opportunity++":
            self.data_root = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/"
            self.json_path = os.path.join(self.data_root, "action")
            self.stats_file_path = "/mnt/hdd4tb/junho/Opportunity++/sensor_stats/sensor_stats_37.npy"
            self.start_index, self.end_index = 194, 230
            self.cache_dir = os.path.join(self.data_root, "caches")

        elif args.dataset_name == "HWU-USP":
            self.data_root = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/"
            self.json_path = "/mnt/hdd4tb/junho/HWU-USP_v2/motion_2_priority_test=18"
            self.stats_file_path = "/mnt/hdd4tb/junho/HWU-USP_v2/sensor_stats_6_with_trashes.npy"
            self.start_index, self.end_index = 4, 9
            self.cache_dir = "/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window/caches"

        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")

        # ============================================================
        # ✅ Stage별 JSON 분기
        # ============================================================
        if stage == 'pretrain':
            print("INFO: DataModule configured for PRE-TRAINING stage.")
            self.json_train_path = os.path.join(self.json_path, "pretrain_cropped_with_flow.json")
            self.json_val_path = (
                os.path.join(self.json_path, "pretrain_cropped_with_flow.json") if args.model_name == "method" else None
            )
            self.json_test_path = None

        elif stage == 'linear_probe':
            print("INFO: DataModule configured for LINEAR PROBE stage.")
            self.json_train_path = os.path.join(self.json_path, "linear_train.json")
            self.json_val_path = os.path.join(self.json_path, "linear_val.json")
            self.json_test_path = os.path.join(self.json_path, "linear_test.json")

        elif stage == 'linear_probe_lstm':
            print("INFO: DataModule configured for LINEAR PROBE LSTM stage.")
            self.json_train_path = os.path.join(self.json_path, "linear_probe_train.json")
            self.json_val_path = os.path.join(self.json_path, "linear_probe_val.json")
            self.json_test_path = os.path.join(self.json_path, "linear_probe_test.json")

        else:
            raise ValueError(f"Invalid stage: {stage}. Choose 'pretrain', 'linear_probe', or 'linear_probe_lstm'.")

    # ============================================================
    # ✅ prepare_data: mean/std 계산 (최초 1회)
    # ============================================================
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
                cache_dir=self.cache_dir,
                use_flow=self.use_flow
            )
            stats = calculate_sensor_stats(temp_dataset)
            save_stats(stats, self.stats_file_path)

    # ============================================================
    # ✅ setup: Dataset 로드 (Stage별 분기)
    # ============================================================
    def setup(self, stage=None):
        stats = load_stats(self.stats_file_path)
        sensor_preprocessor = SensorTransform(
            target_len=128, mean=stats["mean"], std=stats["std"]
        )

        # -------------------------------
        # ① 기존 pretrain / linear_probe
        # -------------------------------
        if self.stage in ["pretrain", "linear_probe"]:
            self.train_dataset = VideoSensorDataset(
                json_path=self.json_train_path,
                data_root=self.data_root,
                num_frames=self.num_frames,
                transform=self.train_transform,
                sensor_transform=sensor_preprocessor,
                threshold_epoch=self.threshold_epoch,
                start_index=self.start_index,
                end_index=self.end_index,
                cache_dir=self.cache_dir,
                use_flow=self.use_flow
            )
            print(f"Train dataset size: {len(self.train_dataset)}")

            if self.json_val_path:
                self.val_dataset = VideoSensorDataset(
                    json_path=self.json_val_path,
                    data_root=self.data_root,
                    num_frames=self.num_frames,
                    transform=self.train_transform,
                    sensor_transform=sensor_preprocessor,
                    threshold_epoch=self.threshold_epoch,
                    start_index=self.start_index,
                    end_index=self.end_index,
                    cache_dir=self.cache_dir,
                    use_flow=self.use_flow
                )

            if self.json_test_path:
                self.test_dataset = VideoSensorDataset(
                    json_path=self.json_test_path,
                    data_root=self.data_root,
                    num_frames=self.num_frames,
                    transform=self.train_transform,
                    sensor_transform=sensor_preprocessor,
                    threshold_epoch=self.threshold_epoch,
                    start_index=self.start_index,
                    end_index=self.end_index,
                    cache_dir=self.cache_dir,
                    use_flow=self.use_flow
                )
                

        # -------------------------------
        # ② LSTM용 시퀀스 데이터셋
        # -------------------------------
        elif self.stage == "linear_probe_lstm":
            print("Loading SequenceDataset for Linear Probe LSTM ...")
            
            # encoder_type에 따라 video 포함 여부 결정
            include_video = getattr(self, 'encoder_type', 'sensor') in ['video', 'sensor-video']
            
            shared_class_to_idx = {}
            self.train_dataset = SequenceDataset(
                json_path=self.json_train_path,
                data_root=self.data_root,
                class_to_idx=shared_class_to_idx,
                sensor_transform=sensor_preprocessor,
                include_video=include_video,
                num_frames=self.num_frames,
                video_transform=self.train_transform if include_video else None,
                cache_dir=os.path.join(self.cache_dir, "videos"),
                use_flow=self.use_flow
            )
            self.val_dataset = SequenceDataset(
                json_path=self.json_test_path,
                data_root=self.data_root,
                class_to_idx=shared_class_to_idx,
                sensor_transform=sensor_preprocessor,
                include_video=include_video,
                num_frames=self.num_frames,
                video_transform=self.train_transform if include_video else None,
                cache_dir=os.path.join(self.cache_dir, "videos"),
                use_flow=self.use_flow
            )
            self.test_dataset = SequenceDataset(
                json_path=self.json_test_path,
                data_root=self.data_root,
                class_to_idx=shared_class_to_idx,
                sensor_transform=sensor_preprocessor,
                include_video=include_video,
                num_frames=self.num_frames,
                video_transform=self.train_transform if include_video else None,
                cache_dir=os.path.join(self.cache_dir, "videos"),
                use_flow=self.use_flow
            )

            self.collate_fn = collate_variable_length

        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def train_dataloader(self):
        if self.stage == "linear_probe_lstm":
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=1,  # ✅ sequence는 1개씩 (video encoding은 batch_size개씩)
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.stage == "linear_probe_lstm":
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=1,  # ✅ sequence는 1개씩
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    def test_dataloader(self):
        """테스트용 DataLoader 반환"""
        if self.stage == "linear_probe_lstm":
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=1,  # ✅ sequence는 1개씩
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    


    # VideoSensorDataset의 set_epoch를 호출하기 위한 콜백
    def on_before_train_epoch(self, epoch):
        self.train_dataset.set_epoch(epoch)