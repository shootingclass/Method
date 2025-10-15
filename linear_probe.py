import os
import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
# comodo는 SVM

# --- ⭐️ 사용자 정의 모듈 임포트 (중요!) ⭐️ ---
# 아래 두 클래스는 사용자님의 다른 파일에서 가져와야 합니다.
# 이 스크립트와 같은 위치에 data.py와 pretrain.py가 있다고 가정합니다.
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


class LinearProbeLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # 1. Pre-trained 모델을 로드합니다.
        pretrain_model = MethodLightningModule.load_from_checkpoint(self.hparams.checkpoint_path)
        
        # 2. Pre-trained 모델에서 '센서 인코더' 부분만 가져옵니다.
        self.sensor_encoder = pretrain_model.sensor_encoder
        
        # 3. (미세조정) 센서 인코더를 학습 가능한 상태(train mode)로 둡니다.
        #    Linear Probing과 달리, Fine-tuning에서는 백본의 파라미터도 함께 학습합니다.
        self.sensor_encoder.train()
            
        # 4. 새로운 선형 분류기(Linear Classifier)만 추가합니다.
        self.classifier = torch.nn.Linear(
            self.sensor_encoder.output_dim, 
            self.hparams.num_classes
        )
        
        # 5. Loss 함수 및 정확도 메트릭 정의
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, sensor_data):
        # 백본(인코더)과 분류기를 순서대로 통과시킵니다.
        # Fine-tuning에서는 백본의 그래디언트도 계산해야 하므로 torch.no_grad()를 사용하지 않습니다.
        representations = self.sensor_encoder(sensor_data)
        logits = self.classifier(representations)
        return logits

    def _shared_step(self, batch, batch_idx):
        _, sensor_data, y, _ = batch
        logits = self(sensor_data)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.val_accuracy.update(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.test_accuracy.update(preds, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)

    def configure_optimizers(self):
        # ⭐️ 가장 중요한 부분: Fine-tuning을 위해 옵티마이저를 두 그룹으로 나눕니다. ⭐️
        # 백본은 작은 learning rate로, 새로 추가된 분류기는 큰 learning rate로 학습합니다.
        param_groups = [
            {'params': self.classifier.parameters(), 'lr': self.hparams.lr},
            {'params': self.sensor_encoder.parameters(), 'lr': self.hparams.backbone_lr}
        ]
        optimizer = torch.optim.Adam(param_groups)
        return optimizer

def main(args):
    set_random_seed(42)

    # 1. 데이터 모듈 초기화 (Linear Probing 단계로 설정)
    datamodule = MethodDataModule(args, stage='linear_probe')

    # 2. 라이트닝 모듈 초기화
    model = LinearProbeLightningModule(args)

    # 3. 로거 및 체크포인트 콜백 설정
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logger = WandbLogger(project="Method_Linear_Probe", name=f"probe_{args.exp_name}") if is_master_process else False
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        dirpath=f'checkpoints_linear/{args.exp_name}',
        filename='best_probe_model-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1
    )

    # 4. 트레이너 설정 및 학습/테스트 시작
    trainer = pl.Trainer(
        max_epochs=args.linear_epochs,
        accelerator='gpu',
        devices=args.devices,
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
    
    # --- 데이터 관련 인자 ---
    parser.add_argument('--dataset_name', type=str, required=True, choices=['Opportunity++', 'HWU-USP'])
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)

    # --- 모델/학습 관련 인자 ---
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes for the final classifier.')
    parser.add_argument('--linear_epochs', type=int, default=50, help='Number of epochs for linear probing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the linear classifier.')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use.')

    # --- 로깅/실험 관리 인자 ---
    parser.add_argument('--exp_name', type=str, default='default_run', help='Experiment name for logging and saving checkpoints.')
    
    args = parser.parse_args()
    main(args)