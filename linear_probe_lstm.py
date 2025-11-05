import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from method_utils import gather
from linear_probe_lstm import LinearProbeLSTMDatamodule


def set_random_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_module_params(args):
    parts = args.checkpoint_path.split('/')
    if len(parts) >= 3:
        args.dataset_name = parts[-2]  # 예: HWU-USP
        args.model_name = parts[-3]    # 예: method/comodo/primus/imu2clip/mae
        args.ckpt_name = parts[-1].split('.')[0]
        print(f"Dataset: {args.dataset_name}, Model: {args.model_name}")
    else:
        print("경고: checkpoint_path 구조가 예상과 다릅니다. (…/MODEL/DATASET/xxx.ckpt)")
    args.mid_label = True
    return args


def load_pretrained_model(args):
    ckpt = args.checkpoint_path
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"--- Loading pretrained model from: {ckpt} ---")

    if args.model_name == "method":
        from method import MethodLightningModule
        model = MethodLightningModule.load_from_checkpoint(ckpt)
    elif args.model_name == "comodo":
        from baseline_modules.comodo.module import COMODOLightningModule
        model = COMODOLightningModule.load_from_checkpoint(ckpt, strict=False, map_location='cpu').to('cuda')
    elif args.model_name == "primus":
        from baseline_modules.primus import PRIMUSLightningModule
        model = PRIMUSLightningModule.load_from_checkpoint(ckpt)
    elif args.model_name == "imu2clip":
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        model = IMU2CLIPLightningModule.load_from_checkpoint(ckpt)
    elif args.model_name == "mae":
        from baseline_modules.mae import CAVMAELightningModule
        model = CAVMAELightningModule.load_from_checkpoint(ckpt, map_location='cpu').to('cuda')
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    return model


class LinearProbeLSTM(pl.LightningModule):
    """
    윈도우 임베딩(사전학습 센서 인코더) → LSTM → 시퀀스(액티비티) 분류
    """
    def __init__(self, args, backbone):
        super().__init__()
        self.save_hyperparameters(args)
        self.backbone = backbone  # 사전학습 모듈(PLModule)

        # 임베딩 차원: method만 *2, 그 외 그대로
        emb_dim = int(self.backbone.hparams.embedding_dim)
        if self.hparams.model_name == "method":
            emb_dim *= 2
        self.emb_dim = emb_dim

        # LSTM: variable-length 시퀀스 입력
        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.emb_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.classifier = nn.Linear(self.emb_dim, self.hparams.num_classes)

        # Metrics
        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')

        self.val_preds_raw = []
        self.val_labels_raw = []
        self.test_preds_raw = []
        self.test_labels_raw = []

        # 백본 고정 (linear probing 성격 유지)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def _encode_windows(self, windows_b_sct):
        """
        windows_b_sct: [B, S, C, T] -> 반환 [B, S, D]
        Backbone의 센서 인코더를 통해 윈도우 임베딩 추출.
        """
        B, S, C, T = windows_b_sct.shape
        flat = windows_b_sct.reshape(B * S, C, T)  # [B*S, C, T]

        # 백본별 forward 규칙
        if self.hparams.model_name in ["method", "comodo", "primus", "imu2clip"]:
            sensor_encoder = self.backbone.sensor_model

        # 모델별 전처리/호출
        if self.hparams.model_name == "method":
            reps = sensor_encoder(flat)  # [B*S, D/2?] -> 실제는 D/2 각각? method는 최종 concat 결과가 D*2임을 가정
        elif self.hparams.model_name == "imu2clip":
            flat_padded = self.backbone.sensor_padding(flat)
            reps = sensor_encoder(flat_padded)  # [B*S, D]
        elif self.hparams.model_name == "primus":
            reps = sensor_encoder(flat)['mmcl']  # [B*S, D]
        elif self.hparams.model_name == "mae":
            reps = self.backbone.model.forward_sensor_only(flat)  # [B*S, D]
        elif self.hparams.model_name == "comodo":
            reps = sensor_encoder(flat)  # [B*S, D]
        else:
            raise ValueError(f"Unknown model_name: {self.hparams.model_name}")

        # method만 *2 차원일 것을 기대 (이미 모델 내부에서 concat되어 나온다면 emb_dim과 일치)
        reps = reps.reshape(B, S, -1)  # [B, S, D_eff]
        return reps

    def forward(self, windows_b_sct, lengths_b):
        """
        windows_b_sct: [B, S, C, T], lengths_b: [B]
        Returns logits: [B, num_classes]
        """
        with torch.no_grad():
            seq_emb = self._encode_windows(windows_b_sct)  # [B, S, D]

        # pack sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_emb, lengths_b.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        # 마지막 layer의 h_n: [num_layers * num_directions, B, H]
        h_last = h_n[-1]  # [B, H]
        logits = self.classifier(h_last)  # [B, num_classes]
        return logits

    def _shared_step(self, batch):
        windows, lengths, targets, metas = batch  # windows:[B,S,C,T]
        windows = windows.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device)
        targets = targets.to(self.device)

        logits = self(windows, lengths)
        loss = self.criterion(logits, targets)
        preds = logits.argmax(dim=1)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_preds_raw.clear()
        self.val_labels_raw.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.val_acc.update(preds, targets)
        self.val_f1_macro.update(preds, targets)
        self.val_preds_raw.append(preds.detach().cpu())
        self.val_labels_raw.append(targets.detach().cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if len(self.val_preds_raw) == 0:
            return
        
        val_labels = gather(torch.cat(self.val_labels_raw, dim=0))
        val_preds = gather(torch.cat(self.val_preds_raw, dim=0))
        y_true = val_labels.cpu().numpy()
        y_pred = val_preds.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)

        # ✅ 클래스 이름 직접 지정
        class_names = ["cereals", "dishes", "sandwich", "tea", "tidy"]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_title("Validation Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        if self.logger:
            self.logger.experiment.log({"val/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)


    def on_test_epoch_start(self):
        self.test_preds_raw.clear()
        self.test_labels_raw.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.test_acc.update(preds, targets)
        self.test_f1_macro.update(preds, targets)
        self.test_preds_raw.append(preds.detach().cpu())
        self.test_labels_raw.append(targets.detach().cpu())
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_f1_macro", self.test_f1_macro, on_epoch=True)

    
    def on_test_epoch_end(self):
        if len(self.test_preds_raw) == 0:
            return
        
        test_labels = gather(torch.cat(self.test_labels_raw, dim=0))
        test_preds = gather(torch.cat(self.test_preds_raw, dim=0))
        y_true = test_labels.cpu().numpy()
        y_pred = test_preds.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)

        # ✅ 동일한 클래스 이름 사용
        class_names = ["cereals", "dishes", "sandwich", "tea", "tidy"]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        if self.logger:
            self.logger.experiment.log({"test/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)


    def configure_optimizers(self):
        # 백본은 freeze, LSTM+classifier만 학습
        params = list(self.lstm.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return optimizer


def main():
    parser = argparse.ArgumentParser()

    # 필수
    parser.add_argument('--checkpoint_path', type=str, required=True)

    # 데이터
    parser.add_argument('--data_root', type=str, default='/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window')
    parser.add_argument('--train_json', type=str, default= '/mnt/hdd4tb/junho/HWU-USP_v2/motion_2_almost_priority/linear_probe_train.json')
    parser.add_argument('--test_json', type=str, default= '/mnt/hdd4tb/junho/HWU-USP_v2/motion_2_almost_priority/linear_probe_test.json')

    # 학습
    parser.add_argument('--num_classes', type=int, default=5)  # HWU-USP 기준
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--linear_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)

    # 기타
    parser.add_argument('--devices', type=int, default=-1)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_true')
    parser.add_argument('--project', type=str, default='Method_Linear_Probe')
    parser.add_argument('--run_name', type=str, default=None)

    args = parser.parse_args()
    set_random_seed(42)
    args = set_module_params(args)

    # Data
    dm = LinearProbeLSTMDatamodule(
        data_root=args.data_root,
        train_json=args.train_json,
        test_json=args.test_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dm.setup()

    # Backbone + Probe
    backbone = load_pretrained_model(args)
    model = LinearProbeLSTM(args, backbone)

    # Logger & Trainer
    is_master = os.environ.get("LOCAL_RANK", "0") == "0"
    if args.run_name == None:
        args.run_name = f"{args.model_name}_{args.dataset_name}_{args.ckpt_name}_last_{args.batch_size*4}_epoch={backbone.hparams.epochs}_linearEpoch={args.linear_epochs}"
    logger = wandb.init(project=args.project, name=args.run_name) if is_master else None
    wb_logger = pl.loggers.WandbLogger(experiment=logger) if logger else False

    ckpt_dir = os.path.join("checkpoints_linear_lstm",
                            f"{args.model_name}_{args.dataset_name}_{args.ckpt_name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=args.linear_epochs,
        accelerator='gpu',
        devices=args.devices,
        strategy=args.strategy,
        logger=wb_logger,
        callbacks=[ckpt_cb]
    )

    print("--- Start Linear Probe (LSTM) ---")
    trainer.fit(model, datamodule=dm)
    print("--- Testing on best checkpoint ---")
    trainer.test(model=None, datamodule=dm, ckpt_path='best')

    if logger:
        wandb.finish()


if __name__ == "__main__":
    main()
