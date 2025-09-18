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


def main(args):
    set_random_seed(42)

    # 1. 데이터 모듈 초기화
    datamodule = MethodDataModule(args)

    # 2. 라이트닝 모듈 초기화
    model = MethodLightningModule(args, datamodule.train_dataloader)

    # 3. 로거 설정 (마스터 프로세스에서만!)
    # LOCAL_RANK 환경 변수를 확인하여 rank 0 프로세스에서만 로거를 생성합니다.
    # rank 0이 아닌 다른 프로세스에서는 logger를 False로 설정하여 로깅을 비활성화합니다.
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logger = WandbLogger(project="Method_Test_Lightning", name="Test1") if is_master_process else False
    wandb.init(project="Method_Test_Lightning", name="Test1") if is_master_process else None

    # 4. 트레이너 설정 및 학습 시작
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        # devices=4,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,  # 여기에 설정된 로거를 전달합니다.
        callbacks=[DatasetEpochCallback()]
    )

    print("--- Starting Training with PyTorch Lightning ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")


####################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Method Test with PyTorch Lightning")
    # 경로 인자
    parser.add_argument("--dataset_name", type=str, default="Opportunity++", help="Dataset name")
    parser.add_argument("--visualize_output_dir", type=str, default="/home/junho/Method/Visualization/transformed_video", help="Directory to save visualization outputs")
    
    # 학습 인자    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--alpha_fixed", type=bool, default=True)
    parser.add_argument("--threshold_epoch", type=int, default=9)
    
    args = parser.parse_args()

    main(args)