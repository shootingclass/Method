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
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

# --- 사용자 정의 모듈 임포트 ---
from datamodule import MethodDataModule
from method import MethodLightningModule

####################################################################

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_model_params(args, datamodule):
    global caching_callback
    if args.dataset_name == "Opportunity++":
        args.num_sensors = 37
        args.num_classes = 7
        args.top_k = 4
        args.sensor_seq_len = 128
    elif args.dataset_name == "HWU-USP":
        args.num_sensors = 11
        args.num_classes = 9
        args.top_k = 1
        args.sensor_seq_len = 100

    args.baseline_video_cache_dir = f"./video_caches/{args.model_name}/{args.dataset_name}"
    if args.model_name == "method":
        return MethodLightningModule(args, datamodule.val_dataloader())
    elif args.model_name == "comodo":
        from baseline_modules.comodo import initialize_comodo
        args.video_ckpt = "facebook/timesformer-base-finetuned-k400"
        args.imu_ckpt = "paris-noah/Mantis-8M"
        args.mlp_hidden_dim = 2048
        args.mlp_output_dim = 128
        args.reduction = "concat" # "mean", "concat"
        return initialize_comodo(args, datamodule)
    elif args.model_name == "primus":
        from baseline_modules.primus import PRIMUSLightningModule
        args.ssl_coeff = 0.5
        args.nnclr = True
        args.transform_indices = [2, 4]
        args.embedding_dim = 512
        return PRIMUSLightningModule(args)
    elif args.model_name == "imu2clip":
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        return IMU2CLIPLightningModule(args)
    elif args.model_name == "mae":
        from baseline_modules.mae import CAVMAELightningModule
        args.masking_ratio = 0.75
        args.contrast_loss_weight = 0.01
        args.mae_loss_weight = 1.0
        args.norm_pix_loss = True
        return CAVMAELightningModule(args)
    else:
        NameError(f"Invalid model name: {args.model_name}")   

####################################################################


# 데이터 모듈의 train_dataloader에서 데이터셋을 가져와 set_epoch 호출
class DatasetEpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("dataset epoch callback called")
        if hasattr(trainer.datamodule.train_dataset, 'set_epoch'):
            trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch)


####################################################################


def main(args):
    set_random_seed(42)

    # 1. 데이터 모듈 초기화
    datamodule = MethodDataModule(args, "pretrain")
    datamodule.setup(stage='fit')

    # 2. 라이트닝 모듈 초기화
    args.cache_dir = datamodule.cache_dir
    model = set_model_params(args, datamodule)
    # model = MethodLightningModule(args, eval_datamodule.train_dataloader)

    # 3. 로거 설정 (마스터 프로세스에서만!)
    # LOCAL_RANK 환경 변수를 확인하여 rank 0 프로세스에서만 로거를 생성합니다.
    # rank 0이 아닌 다른 프로세스에서는 logger를 False로 설정하여 로깅을 비활성화합니다.
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logger = WandbLogger(project="Method_Test_Lightning", name="Test1") if is_master_process else False
    wandb.init(project="Method_Test_Lightning", name="Test1") if is_master_process else None
    
    # 4. 콜백 리스트 생성
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{args.model_name}/{args.dataset_name}",  # 모델이 저장될 폴더
        filename="pretrained_model-{epoch:02d}-{val_loss:.2f}", # 저장될 파일 이름 형식
        save_top_k=1,            # 가장 좋은 모델 1개만 저장
        monitor="val_loss",      # val_loss를 기준으로 성능을 판단
        mode="min",              # val_loss는 낮을수록 좋으므로 'min' 모드
    )
    callbacks = [DatasetEpochCallback(), checkpoint_callback]

    # 4. 트레이너 설정 및 학습 시작
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        # devices=4,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,  # 여기에 설정된 로거를 전달합니다.
        callbacks=callbacks
    )

    print("--- Starting Training with PyTorch Lightning ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")


####################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Method Test with PyTorch Lightning")
    # 경로 인자
    parser.add_argument("--dataset_name", type=str, default="Opportunity++", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="method")
    parser.add_argument("--visualize_output_dir", type=str, default="/home/junho/Method/Visualization/transformed_video", help="Directory to save visualization outputs")
    
    # 학습 인자    
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--alpha_fixed", type=bool, default=True)
    parser.add_argument("--threshold_epoch", type=int, default=9)
    
    args = parser.parse_args()
    args.stage = "pretrain"

    main(args)