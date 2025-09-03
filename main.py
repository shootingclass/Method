import os
import numpy as np
import random
import itertools
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

# --- 사용자 정의 모듈 임포트 ---
from datamodule import MethodDataModule
from lightning_module import MethodLightningModule


####################################################################


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


####################################################################


# 데이터 모듈의 train_dataloader에서 데이터셋을 가져와 set_epoch 호출
class DatasetEpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(trainer.datamodule.train_dataset, 'set_epoch'):
            trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch)


####################################################################


torch.backends.cuda.preferred_linalg_library("magma") 


def main(args):
    set_random_seed(42)

    # 1. 데이터 모듈 초기화
    datamodule = MethodDataModule(args)

    # 2. 라이트닝 모듈 초기화
    model = MethodLightningModule(args, datamodule.train_dataloader)

    # 3. 로거 및 wandb 설정
    wandb_logger = WandbLogger(project="Method_Test_Lightning", name="Test1")
    wandb.init(project="Method_Test_Lightning", name="Test1")

    # 4. 트레이너 설정 및 학습 시작
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=[1, 2, 3],
        strategy='ddp_find_unused_parameters_true',
        logger=wandb_logger,
        callbacks=[DatasetEpochCallback()]
    )

    print("--- Starting Training with PyTorch Lightning ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")


####################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Method Test with PyTorch Lightning")

    # 경로 인자
    parser.add_argument("--data_root", type=str, default="/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/", help="Root directory of the dataset")
    parser.add_argument("--json_train_path", type=str, default="/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/actionOnlyObject/pretrain.json", help="Path to the training JSON file")
    parser.add_argument("--stats_file_path", type=str, default='/mnt/hdd4tb/junho/Opportunity++/sensor_stats/sensor_stats_37.npy', help="Path to the sensor stats file")
    parser.add_argument("--visualize_output_dir", type=str, default="/home/junho/Method/Visualization/transformed_video", help="Directory to save visualization outputs")
    
    # 학습 인자    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--alpha_fixed", type=bool, default=True)
    parser.add_argument("--num_sensors", type=int, default=37)
    parser.add_argument("--threshold_epoch", type=int, default=9)
    
    args = parser.parse_args()

    main(args)