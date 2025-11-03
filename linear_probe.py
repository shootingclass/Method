import os
import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix

# --- ⭐️ 사용자 정의 모듈 임포트 (중요!) ⭐️ ---
# 아래 두 클래스는 사용자님의 다른 파일에서 가져와야 합니다.
# 이 스크립트와 같은 위치에 data.py와 pretrain.py가 있다고 가정합니다.
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

def set_module_params(args):
    parts = args.checkpoint_path.split('/') 
    # parts = ['.', 'checkpoints', 'CAVMAE', 'HWU-USP', 'pretrained_model.ckpt']

    # 뒤에서 세면: -1은 파일명, -2는 데이터셋, -3은 모델명
    if len(parts) >= 3:
        args.dataset_name = parts[-2] # HWU-USP
        args.model_name = parts[-3]   # CAVMAE
        args.ckpt_name = parts[-1].split('.')[0]
        print(f"Dataset: {args.dataset_name}, Model: {args.model_name}")
    else:
        print("경로 구조가 예상과 다릅니다.")
    return args

def set_linear_probe_module(args):
    if args.dataset_name == "Opportunity++":
        args.num_classes = 14
    elif args.dataset_name == "HWU-USP":
        args.num_classes = 10
    else:
        ValueError(f"{args.dataset_name} is not valid dataset")
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found or not specified: {checkpoint_path}")

    print(f"--- Loading pre-trained model from: {checkpoint_path} ---")

    if args.model_name == "method":
        # Import moved inside
        from method import MethodLightningModule # Replace with actual name
        model = MethodLightningModule.load_from_checkpoint(
            checkpoint_path,
        )
    elif args.model_name == "comodo":
        # Import moved inside
        from baseline_modules.comodo.module import COMODOLightningModule # Assuming this is the correct class
        model = COMODOLightningModule.load_from_checkpoint(
            checkpoint_path,
            strict = False, # Comodo Loss 는 정의되지 않음.
            map_location='cpu'
        )
        model = model.to('cuda')
    elif args.model_name == "primus":
        # Import moved inside
        from baseline_modules.primus import PRIMUSLightningModule
        model = PRIMUSLightningModule.load_from_checkpoint(
            checkpoint_path,
        )
    elif args.model_name == "imu2clip":
        # Import moved inside
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        model = IMU2CLIPLightningModule.load_from_checkpoint(
            checkpoint_path,
        )
    elif args.model_name == "mae": # sync?
        # Import moved inside
        from baseline_modules.mae import CAVMAELightningModule
        model = CAVMAELightningModule.load_from_checkpoint(
            checkpoint_path, map_location='cpu'
        )
        model = model.to('cuda')
    else:
        raise ValueError(f"Unknown model_name for loading: {args.model_name}")
    return LinearProbeLightningModule(args, model)

####################################################################


class LinearProbeLightningModule(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.save_hyperparameters(args)

        # 1. 뒤에 모델 내부 함수 혹은 하이퍼 파라미터 사용을 위해 모델 저장
        self.model = model
        
        # 3. (미세조정) 센서 인코더를 학습 가능한 상태(train mode)로 둡니다.
        #    Linear Probing과 달리, Fine-tuning에서는 백본의 파라미터도 함께 학습합니다.
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False  # gradient 막기
            
        # 4. 새로운 선형 분류기(Linear Classifier)만 추가합니다.
        self.classifier = torch.nn.Linear(
            model.hparams.embedding_dim *2,
            # 128,
            # model.hparams.mlp_output_dim,
            # 768,
            self.hparams.num_classes
        )
        
        # 5. Loss 함수 및 정확도 메트릭 정의
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        # F1 Scores (Micro, Macro, Weighted)
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='micro')
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.val_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='weighted')
        
        self.test_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='micro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='weighted')

        # mAUC (Multiclass AUROC) - average='macro'가 일반적
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.hparams.num_classes, average='macro')

        # mAP (Multiclass Average Precision) - average='macro'가 일반적
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=self.hparams.num_classes, average='macro')

        # --- ✅ Confusion Matrix 추가 ---
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

        # wandb 로깅을 위한 클래스 이름 (예시)
        self.class_dic = {0: 'Open Door 1',
            1: 'Open Door 2',
            2: 'Close Door 1',
            3: 'Close Door 2',
            4: 'Open Fridge',
            5: 'Close Fridge',
            6: 'Open Dishwasher',
            7: 'Close Dishwasher',
            8: 'Open Drawer 1',
            9: 'Close Drawer 1',
            10: 'Open Drawer 2',
            11: 'Close Drawer 2',
            12: 'Open Drawer 3',
            13: 'Close Drawer 3',
         }
        self.class_names = [v for v in self.class_dic.values()]

    def forward(self, sensor_data):
        # 백본(인코더)과 분류기를 순서대로 통과시킵니다.
        # Fine-tuning에서는 백본의 그래디언트도 계산해야 하므로 torch.no_grad()를 사용하지 않습니다.
        if self.hparams.model_name == "method":
            sensor_encoder = self.model.sensor_model
            representations = sensor_encoder(sensor_data)
        elif self.hparams.model_name == "imu2clip":
            sensor_encoder = self.model.sensor_model
            sensor_data = self.model.sensor_padding(sensor_data)
            representations = sensor_encoder(sensor_data)
        elif self.hparams.model_name == "primus":
            sensor_encoder = self.model.sensor_model
            representations = sensor_encoder(sensor_data)['mmcl']
        elif self.hparams.model_name == "mae":
            representations = self.model.model.forward_sensor_only(sensor_data)
        elif self.hparams.model_name == "comodo":
            sensor_encoder = self.model.sensor_model
            representations = sensor_encoder(sensor_data)
        else:
            raise ValueError(f"Unknown model_name for loading: {self.hparams.model_name}")
            
        logits = self.classifier(representations)
        return logits

    def _shared_step(self, batch, batch_idx):
        _, sensor_data, y, _, _ = batch
        logits = self(sensor_data)
        loss = self.criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, probs, y

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch, batch_idx)
        self.val_accuracy.update(preds, y)
        self.val_f1_micro.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_f1_weighted.update(preds, y)
        self.val_auroc.update(probs, y) # AUC는 확률값(probs) 사용
        self.val_ap.update(probs, y)    # AP는 확률값(probs) 사용
        # ✅ confusion matrix용 raw 저장
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(y.detach().cpu())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1_micro", self.val_f1_micro, on_step=False, on_epoch=True, prog_bar=False) # prog_bar는 선택사항
        self.log("val_f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1_weighted", self.val_f1_weighted, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_mAUC", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mAP", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

        
    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_labels).cpu().numpy()
        y_pred = torch.cat(self.val_preds).cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)  # 정규화 버전 (선택)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,                     # raw count
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True, square=True,
            ax=ax
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Validation Confusion Matrix")

        # W&B에 업로드
        if self.logger is not None:
            self.logger.experiment.log({"Validation Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []

    def test_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch, batch_idx)
        self.test_accuracy.update(preds, y)
        self.test_f1_micro.update(preds, y)
        self.test_f1_macro.update(preds, y)
        self.test_f1_weighted.update(preds, y)
        self.test_auroc.update(probs, y)
        self.test_ap.update(probs, y)

        # ✅ raw 저장
        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)
        self.log("test_f1_macro", self.test_f1_macro, on_epoch=True)
        self.log("test_mAUC", self.test_auroc, on_epoch=True)
        self.log("test_mAP", self.test_ap, on_epoch=True)

    def on_test_epoch_end(self):
        y_true = torch.cat(self.test_labels).cpu().numpy()
        y_pred = torch.cat(self.test_preds).cpu().numpy()

          # ✅ DDP 상태에서 모든 GPU의 결과를 모음
        # gathered_true = [torch.zeros_like(y_true) for _ in range(self.trainer.world_size)]
        # gathered_pred = [torch.zeros_like(y_pred) for _ in range(self.trainer.world_size)]
        # torch.distributed.all_gather(gathered_true, y_true)
        # torch.distributed.all_gather(gathered_pred, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Test Confusion Matrix")
        if self.logger is not None:
            self.logger.experiment.log({"Test Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)


    def configure_optimizers(self):
        # ⭐️ 가장 중요한 부분: Fine-tuning을 위해 옵티마이저를 두 그룹으로 나눕니다. ⭐️
        # 백본은 작은 learning rate로, 새로 추가된 분류기는 큰 learning rate로 학습합니다.
        param_groups = [
            {'params': self.classifier.parameters(), 'lr': self.hparams.lr},
            {'params': self.model.parameters(), 'lr': self.hparams.backbone_lr}
        ]
        optimizer = torch.optim.Adam(param_groups)
        return optimizer

def main(args):
    set_random_seed(42)

    args = set_module_params(args)

    # 1. 데이터 모듈 초기화 (Linear Probing 단계로 설정)
    datamodule = MethodDataModule(args, stage='linear_probe')

    # 2. 라이트닝 모듈 초기화
    model = set_linear_probe_module(args)

    # 3. 로거 및 체크포인트 콜백 설정
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    save_name = f"{args.model_name}_{args.dataset_name}_{args.ckpt_name}_last_{args.batch_size*4}_epoch={model.model.hparams.epochs}_linearEpoch={args.linear_epochs}"
    logger = WandbLogger(project="Method_Linear_Probe", name=f"probe_{save_name}") if is_master_process else False
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        dirpath=f'checkpoints_linear/{save_name}',
        filename='best_probe_model-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1
    )

    # 4. 트레이너 설정 및 학습/테스트 시작
    trainer = pl.Trainer(
        max_epochs=args.linear_epochs,
        accelerator='gpu',
        devices=-1,
        # devices=args.devices,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    print("--- Starting Linear Probing ---")
    trainer.fit(model, datamodule)
    
    print("\n--- Starting Testing on Best Model ---")
    # fit()이 끝나면 자동으로 최적의 체크포인트가 로드됩니다.
    trainer.test(datamodule=datamodule, ckpt_path='best')
    print("--- Linear Probing and Testing Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- 필수 인자 ---
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the pretrained model checkpoint (.ckpt).')
    # parser.add_argument('--model_name', type=str, required=True)
    
    # --- 데이터 관련 인자 ---
    # parser.add_argument('--dataset_name', type=str, default='Opportunity++', required=True, choices=['Opportunity++', 'HWU-USP'])
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    # --- 모델/학습 관련 인자 ---
    parser.add_argument('--linear_epochs', type=int, default=50, help='Number of epochs for linear probing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the linear classifier.')
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--threshold_epoch', type=int, default=100, help='No need video data, so set threshold higher.')

    # --- 로깅/실험 관리 인자 ---
    parser.add_argument('--exp_name', type=str, default='default_run', help='Experiment name for logging and saving checkpoints.')
    
    args = parser.parse_args()
    main(args)