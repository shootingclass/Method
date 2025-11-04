import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix

from datamodule_lstm import MethodDataModuleLSTM


def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_module_params(args):
    # ckpt 경로에서 모델/데이터셋명 추출(폴더명 규약 가정)
    parts = args.checkpoint_path.split('/')
    if len(parts) >= 3:
        args.dataset_name = parts[-2]
        args.model_name = parts[-3]
        args.ckpt_name = parts[-1].split('.')[0]
    else:
        args.dataset_name = getattr(args, "dataset_name", "HWU-USP")
        args.model_name = getattr(args, "model_name", "method")
        args.ckpt_name = "unknown_ckpt"
    print(f"[Info] Dataset: {args.dataset_name}, Model: {args.model_name}, CKPT: {args.ckpt_name}")
    return args


def build_label_map_from_json(train_json_path):
    with open(train_json_path, 'r') as f:
        obj = json.load(f)
    classes = []
    for seq in obj["data"]:
        # 시퀀스 내 class_name 일관성 체크
        cands = {w["class_name"] for w in seq["windows"]}
        if len(cands) != 1:
            raise ValueError(f"class_name mismatch in sequence {seq.get('sequence_id','<noid>')} -> {cands}")
        classes.append(list(cands)[0])
    classes = sorted(list(set(classes)))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    return class_to_idx


class LinearProbeLSTMLightningModule(pl.LightningModule):
    def __init__(self, args, backbone_model, class_to_idx):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = backbone_model  # pretrained (sensor encoder 포함)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v:k for k,v in class_to_idx.items()}
        num_classes = len(class_to_idx)

        # --- Freeze backbone (Linear Probe) ---
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # --- sensor embedding dim 추출 가이드 ---
        # method 계열: self.model.hparams.embedding_dim (혹은 model.sensor_embedding_dim)
        # 여기선 안정적으로 getattr 사용
        emb_dim = getattr(self.model.hparams, "embedding_dim", None)
        if emb_dim is None:
            # 백업 경로: 모듈 내부 속성 이름이 다를 수 있음 → 조정
            emb_dim = getattr(self.model, "embedding_dim", None)
        if emb_dim is None:
            raise ValueError("Cannot find embedding_dim from the loaded backbone model.")

        self.emb_dim = emb_dim

        # --- LSTM head ---
        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hparams.lstm_hidden_dim,
            num_layers=self.hparams.lstm_num_layers,
            batch_first=True,
            bidirectional=self.hparams.lstm_bidirectional
        )
        lstm_out_dim = self.hparams.lstm_hidden_dim * (2 if self.hparams.lstm_bidirectional else 1)

        self.classifier = nn.Linear(lstm_out_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []

        self.class_names = [self.idx_to_class[i] for i in range(num_classes)]

    def sensor_encode_batch(self, x_btcl):
        """
        x_btcl: [B, T, C, L] float
        return: embeddings [B, T, E]
        """
        B, T, C, L = x_btcl.shape
        x = x_btcl.reshape(B*T, C, L)

        with torch.no_grad():
            if self.hparams.model_name in ["method", "imu2clip", "primus", "comodo"]:
                sensor_encoder = getattr(self.model, "sensor_model", None)
                if sensor_encoder is None:
                    raise ValueError("Backbone has no attribute 'sensor_model'.")
                # 특정 모델은 padding/정규화 유틸이 있을 수 있음 → 필요한 경우 적용
                if hasattr(self.model, "sensor_padding"):
                    x = self.model.sensor_padding(x)
                emb = sensor_encoder(x)  # [B*T, E] 혹은 dict
                if isinstance(emb, dict):
                    # primus 예시: {'mmcl': [B*T, E], ...}
                    emb = emb.get("mmcl", None)
                    if emb is None:
                        raise ValueError("Embedding dict does not contain 'mmcl'.")
            elif self.hparams.model_name in ["mae"]:
                emb = self.model.model.forward_sensor_only(x)  # [B*T, E]
            else:
                raise ValueError(f"Unknown model_name: {self.hparams.model_name}")

        emb = emb.reshape(B, T, -1)
        return emb

    def forward(self, batch):
        """
        batch['sensor_seq']: [B, T, C, L] (pad 포함)
        batch['lengths']:    [B]  유효 길이
        """
        x_btcl = batch["sensor_seq"]           # [B, T, C, L]
        lengths = batch["lengths"]             # [B]

        emb_bte = self.sensor_encode_batch(x_btcl)  # [B, T, E]

        # pack → LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            emb_bte, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)

        # 대표 벡터: 마지막 layer의 마지막 hidden (양방향이면 concat)
        if self.hparams.lstm_bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
        else:
            last_hidden = h_n[-1]  # [B, H]

        logits = self.classifier(last_hidden)  # [B, num_classes]
        return logits

    def _shared_step(self, batch):
        logits = self(batch)
        y = batch["label"]  # [B]
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        return loss, preds, probs, y

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch)
        self.val_accuracy.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_preds.append(preds.cpu()); self.val_labels.append(y.cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_labels).numpy()
        y_pred = torch.cat(self.val_preds).numpy()
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_title("Validation Confusion Matrix")
        if self.logger is not None:
            self.logger.experiment.log({"val_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        self.val_preds, self.val_labels = [], []

    def test_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch)
        self.test_accuracy.update(preds, y)
        self.test_f1_macro.update(preds, y)
        self.test_preds.append(preds.cpu()); self.test_labels.append(y.cpu())
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)
        self.log("test_f1_macro", self.test_f1_macro, on_epoch=True)

    def on_test_epoch_end(self):
        y_true = torch.cat(self.test_labels).numpy()
        y_pred = torch.cat(self.test_preds).numpy()
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_title("Test Confusion Matrix")
        if self.logger is not None:
            self.logger.experiment.log({"test_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        self.test_preds, self.test_labels = [], []

    def configure_optimizers(self):
        # backbone은 freeze, LSTM+classifier만 학습
        params = list(self.lstm.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return optimizer


def load_backbone(args):
    ckpt = args.checkpoint_path
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[Load] {ckpt}")
    if args.model_name == "method":
        from method import MethodLightningModule
        model = MethodLightningModule.load_from_checkpoint(ckpt, map_location='cpu')
    elif args.model_name == "comodo":
        from baseline_modules.comodo.module import COMODOLightningModule
        model = COMODOLightningModule.load_from_checkpoint(ckpt, strict=False, map_location='cpu')
    elif args.model_name == "primus":
        from baseline_modules.primus import PRIMUSLightningModule
        model = PRIMUSLightningModule.load_from_checkpoint(ckpt, map_location='cpu')
    elif args.model_name == "imu2clip":
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        model = IMU2CLIPLightningModule.load_from_checkpoint(ckpt, map_location='cpu')
    elif args.model_name == "mae":
        from baseline_modules.mae import CAVMAELightningModule
        model = CAVMAELightningModule.load_from_checkpoint(ckpt, map_location='cpu')
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    return model


def main(args):
    set_random_seed(42)
    args = set_module_params(args)

    # 클래스 맵
    class_to_idx = build_label_map_from_json(args.train_json)

    # 백본 로드
    backbone = load_backbone(args)

    # DataModule
    dm = MethodDataModuleLSTM(
        args=args,
        train_json=args.train_json,
        test_json=args.test_json,
        processed_root=args.processed_root,
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 모델
    model = LinearProbeLSTMLightningModule(args, backbone, class_to_idx)

    # 로거/CKPT
    is_master = os.environ.get("LOCAL_RANK", "0") == "0"
    save_name = f"{args.model_name}_{args.dataset_name}_{args.ckpt_name}_LSTM_bs{args.batch_size}"
    logger = WandbLogger(project="Method_Linear_Probe_LSTM", name=f"probe_{save_name}") if is_master else False
    ckpt_cb = ModelCheckpoint(
        monitor='val_acc', mode='max',
        dirpath=f'checkpoints_linear_lstm/{save_name}',
        filename='best-{epoch:02d}-{val_acc:.3f}', save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=args.linear_epochs,
        accelerator='gpu', devices=-1,
        strategy='ddp_find_unused_parameters_true',
        logger=logger, callbacks=[ckpt_cb]
    )

    print("\n--- Start LSTM Linear Probing ---")
    trainer.fit(model, datamodule=dm)

    print("\n--- Testing Best ---")
    trainer.test(datamodule=dm, ckpt_path='best')


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # 필수 경로
    p.add_argument('--checkpoint_path', type=str, required=True)
    p.add_argument('--train_json', type=str, default='/mnt/hdd4tb/junho/HWU-USP_v2/motion_2_almost_priority/linear_probe_train.json')
    p.add_argument('--test_json',  type=str, default='/mnt/hdd4tb/junho/HWU-USP_v2/motion_2_almost_priority/linear_probe_test.json')
    p.add_argument('--processed_root', type=str, default='/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window')

    # 학습/모델
    p.add_argument('--model_name', type=str, default='method')  # method/comodo/primus/imu2clip/mae
    p.add_argument('--linear_epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)

    # LSTM 설정
    p.add_argument('--lstm_hidden_dim', type=int, default=256)
    p.add_argument('--lstm_num_layers', type=int, default=1)
    p.add_argument('--lstm_bidirectional', action='store_true', default=True)

    # 로딩/로깅
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=8)

    args = p.parse_args()
    main(args)
