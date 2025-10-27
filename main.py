import os
import numpy as np
import random
import argparse

import torch
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

def set_model(args, datamodule):
    global caching_callback
    if args.dataset_name == "Opportunity++":
        args.num_sensors = 37
        args.num_classes = 7
        args.top_k = 4
        args.sensor_seq_len = 128
        args.min_cluster_size = 100
    elif args.dataset_name == "HWU-USP":
        args.num_sensors = 11
        args.num_classes = 9
        args.top_k = 1
        args.sensor_seq_len = 128
        args.min_cluster_size = 30

    args.baseline_video_cache_dir = f"./video_caches/{args.model_name}/{args.dataset_name}"
    args.seed = 42
    if args.model_name == "method":
        return MethodLightningModule(args, datamodule)
    elif args.model_name == "comodo":
        from baseline_modules.comodo import initialize_comodo
        args.video_ckpt = "facebook/timesformer-base-finetuned-k400"
        args.imu_ckpt = "paris-noah/Mantis-8M"
        args.mlp_hidden_dim = 2048
        args.embedding_dim = 128
        args.queue_size_ratio = 0.1
        args.reduction = "concat" # "mean", "concat"
        args.teacher_temp = 0.1
        args.student_temp = 0.05
        args.learning_rate = 3e-4
        # datamodule.fit("")
        return initialize_comodo(args, datamodule)
    elif args.model_name == "primus":
        from baseline_modules.primus import PRIMUSLightningModule  
        args.num_views = 2 
        args.ssl_coeff = 0.5
        args.nnclr = True
        # args.transform_indices = [2, 4]
        # 원본은 2,4이나 현재 64*4>128 이므로 4에서 에러남. 복원 추출 가능케 하거나 num_segments를 바꾸는 등의 파라미터 수정이 필요.
        args.transform_indices = [0, 1]
        args.embedding_dim = 512
        return PRIMUSLightningModule(args)
    elif args.model_name == "imu2clip":
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        args.embedding_dim = 512
        args.sensor_target_len = 150
        return IMU2CLIPLightningModule(args)
    elif args.model_name == "mae":
        from baseline_modules.mae import CAVMAELightningModule
        args.masking_ratio = 0.75
        args.contrast_loss_weight = 0.01
        args.mae_loss_weight = 1.0
        args.norm_pix_loss = True
        args.lrscheduler_start = 10
        args.lrscheduler_decay = 0.5
        args.lrscheduler_step = 5
        args.embedding_dim = 768 # VIT
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
    # datamodule.setup(stage='fit')

    # 2. 라이트닝 모듈 초기화
    args.cache_dir = datamodule.cache_dir
    model = set_model(args, datamodule)

    # 3. 로거 설정 (마스터 프로세스에서만!)
    # LOCAL_RANK 환경 변수를 확인하여 rank 0 프로세스에서만 로거를 생성합니다.
    # rank 0이 아닌 다른 프로세스에서는 logger를 False로 설정하여 로깅을 비활성화합니다.
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logging_name = f"pretraining_{args.model_name}_{args.dataset_name}"
    logger = WandbLogger(project=args.project_name, name=logging_name) if is_master_process else False
    wandb.init(project=args.project_name, name=logging_name) if is_master_process else None
    # Define correlation metrics
    if is_master_process:
        wandb.define_metric("loss/attn_reg", step_metric="trainer/global_step")
        wandb.define_metric("attention/door_focus_ratio", step_metric="trainer/global_step")
        wandb.define_metric("attention/global_var", step_metric="trainer/global_step")
        wandb.define_metric("attention/door1/diff_mean", step_metric="trainer/global_step")

        # Make a correlation panel artifact (optional but pretty)
        if wandb.run is not None:
            panel = wandb.plot.scatter(
                table=wandb.Table(
                    columns=["door_focus_ratio", "attn_reg"],
                    data=[]
                ),
                x="door_focus_ratio",
                y="attn_reg",
                title="Attention Regularization vs Door Focus",
            )
            wandb.log({"dashboard/attn_correlation": panel})
            
    # 4. 콜백 리스트 생성
    if args.model_name != "method":
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/{args.model_name}/{args.dataset_name}",  # 모델이 저장될 폴더
            filename="pretrained_model-{epoch:02d}-{train_loss:.2f}", # 저장될 파일 이름 형식
            save_top_k=1,            # 가장 좋은 모델 1개만 저장
            monitor="train_loss",      # val_loss를 기준으로 성능을 판단
            mode="min",              # val_loss는 낮을수록 좋으므로 'min' 모드
            save_last=True,
        )
        callbacks = [DatasetEpochCallback(), checkpoint_callback]
    else:
        callbacks = [DatasetEpochCallback()]

    # 4. 트레이너 설정 및 학습 시작
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,  # 여기에 설정된 로거를 전달합니다.
        callbacks=callbacks,
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
    parser.add_argument("--project_name", type=str, default="Method_Test_Lightning", help="WandB project name")

    # 학습 인자    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--alpha_fixed", type=bool, default=True)
    parser.add_argument("--threshold_epoch", type=int, default=5)
    parser.add_argument("--centroid_threshold", type=float, default=0.75)
    parser.add_argument("--guide_start_epoch", type=int, default=5)
    
    args = parser.parse_args()
    args.stage = "pretrain"

    main(args)