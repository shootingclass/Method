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
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

# --- 사용자 정의 모듈 임포트 ---
from datamodule import MethodDataModule
from method import MethodLightningModule
from method_utils import gather
from dataset_lstm import LinearProbeLSTMDatamodule


####################################################################
#                         Utility Functions
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
    if len(parts) >= 3:
        args.dataset_name = parts[-2]
        args.model_name = parts[-3]
        args.ckpt_name = parts[-1].split('.')[0]
        print(f"Dataset: {args.dataset_name}, Model: {args.model_name}")
    else:
        print("경로 구조가 예상과 다릅니다.")
    return args

def get_backbone_with_mode(args):
    if args.supervision:
            print("⚙️  Sensor-only supervised mode enabled (no pretrained weights).")
            
            # 1️⃣ checkpoint의 하이퍼파라미터만 가져오기
            ckpt = torch.load(args.checkpoint_path, map_location="cpu")
            hparams = ckpt["hyper_parameters"] if "hyper_parameters" in ckpt else {}

            # 2️⃣ 기존 load_pretrained_model의 인자를 동일하게 써서 구조만 초기화
            backbone = load_pretrained_model(args, reset_sensor_weights=True)

            # sensor encoder만 학습
            for name, p in backbone.named_parameters():
                print(name)
                # p.requires_grad = "sensor_model" in name
                p.requires_grad = True
            backbone.train()

    else:
        # 기존 pretrained checkpoint 로드 루틴
        backbone = load_pretrained_model(args)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
    return backbone


def load_pretrained_model(args, reset_sensor_weights=False):
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
    
    # 2️⃣ sensor encoder만 weight 초기화 (sensor-only 학습 모드)
    # if reset_sensor_weights:
    #     print("⚙️  Resetting sensor encoder weights for supervised fine-tuning")

    #     sensor_encoder = model.sensor_model
    #     for layer in sensor_encoder.modules():
    #         if hasattr(layer, 'reset_parameters'):
    #             layer.reset_parameters()
    return model


####################################################################
#                        Linear Probe Module
####################################################################

class LinearProbeLightningModule(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model

        # for param in self.model.parameters():
        #     param.requires_grad = False

        emb_dim = model.hparams.embedding_dim
        if self.hparams.model_name == "method":
            emb_dim *= 2
        elif self.hparams.model_name == "comodo":
            emb_dim //= 2

        self.classifier = torch.nn.Linear(emb_dim, self.hparams.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
 # 5. Loss 함수 및 정확도 메트릭 정의
 
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


        # Confusion matrix용 클래스 이름
        self.class_dic = {
            0: 'Open Door 1', 1: 'Open Door 2', 2: 'Close Door 1', 3: 'Close Door 2',
            4: 'Open Fridge', 5: 'Close Fridge', 6: 'Open Dishwasher', 7: 'Close Dishwasher',
            8: 'Open Drawer 1', 9: 'Close Drawer 1', 10: 'Open Drawer 2',
            11: 'Close Drawer 2', 12: 'Open Drawer 3', 13: 'Close Drawer 3'
        }
        self.class_names = [v for v in self.class_dic.values()]

    def forward(self, sensor_data):
        if self.hparams.model_name == "method":
            sensor_encoder = self.model.sensor_model
            representations = sensor_encoder(sensor_data)
            # z_sensor embedding
            # _, features, _ = self.model.clustering_module(sensor_data, return_features = True)
            # sensor_motion_emb = self.model.momentum_sensor_model.encoding_motion(sensor_data)["emb"]
            # import torch.nn.functional as F
            # s_app_norm = F.normalize(features, dim=1)
            # s_mot_norm = F.normalize(sensor_motion_emb, dim=1)
            # # features는 그래디언트 차단
            # representations = torch.cat((s_app_norm.detach(), s_mot_norm.detach()), dim=1)

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
        # ✅ gradient 흐름 확인 (1회만 출력)
        if self.current_epoch == 0:
            if  self.hparams.model_name == "mae":
                encoder = self.model.model
            else:
                encoder = self.model.sensor_model
            grad_flags = [
                (n, p.requires_grad, p.grad is not None)
                for n, p in encoder.named_parameters()
            ]
            trainable = [n for n, rg, _ in grad_flags if rg]
            print(f"[Gradient Check] Trainable params in sensor_encoder: {len(trainable)}")
            total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            print(f"Total trainable parameters in sensor_encoder: {total_params:,}")
            print(f"→ Sample trainable layers: {trainable[:5]}")
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

        self.log("val/val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val/val_f1_micro", self.val_f1_micro, on_step=False, on_epoch=True, prog_bar=False) # prog_bar는 선택사항
        self.log("val/val_f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_f1_weighted", self.val_f1_weighted, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/val_mAUC", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_mAP", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

        
    def on_validation_epoch_end(self):
        val_labels = gather(torch.cat(self.val_labels, dim=0))
        val_preds = gather(torch.cat(self.val_preds, dim=0))
        y_true = val_labels.cpu().numpy()
        y_pred = val_preds.cpu().numpy()

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

        self.log("test/test_loss", loss, on_epoch=True)
        self.log("test/test_acc", self.test_accuracy, on_epoch=True)
        self.log("test/test_f1_macro", self.test_f1_macro, on_epoch=True)
        self.log("test/test_f1_micro", self.test_f1_micro, on_step=False, on_epoch=True, prog_bar=False) # prog_bar는 선택사항
        self.log("test/test_f1_weighted", self.test_f1_weighted, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/test_mAUC", self.test_auroc, on_epoch=True)
        self.log("test/test_mAP", self.test_ap, on_epoch=True)

    def on_test_epoch_end(self):
        test_labels = gather(torch.cat(self.test_labels, dim=0))
        test_preds = gather(torch.cat(self.test_preds, dim=0))
        y_true = test_labels.cpu().numpy()
        y_pred = test_preds.cpu().numpy()

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
        ]
        optimizer = torch.optim.Adam(param_groups)
        return optimizer

class LinearProbeLSTM(pl.LightningModule):
    """
    윈도우 임베딩(사전학습 센서 인코더) → LSTM → 시퀀스(액티비티) 분류
    """
    def __init__(self, args, backbone):
        super().__init__()
        self.save_hyperparameters(args)
        self.backbone = backbone  # 사전학습 모듈(PLModule)
        self.probe_mode = getattr(args, "probe_mode", "lstm")  # 기본값 lstm

        # 임베딩 차원: method만 *2, 그 외 그대로
        emb_dim = int(self.backbone.hparams.embedding_dim)
        if self.hparams.model_name == "method":
            emb_dim *= 2
            # self.backbone.hprams.embedding *=2
        self.emb_dim = emb_dim

             # 🔹 probe_mode가 LSTM일 때만 LSTM 정의
        if self.probe_mode == "lstm":
            self.temporal_model = nn.LSTM(
                input_size=self.emb_dim,
                hidden_size=self.emb_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                dropout=0.3
            )
        elif self.probe_mode == "meanpool":
            self.temporal_model = None  # 단순 mean pooling 사용
        else:
            raise ValueError(f"Unknown probe_mode: {self.probe_mode}")

        # 교체 부분만 발췌
        # self.rnn = nn.GRU(
        #     input_size=self.emb_dim,
        #     hidden_size=self.emb_dim // 2,  # 더 약하게
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=False,
        #     dropout=0.2,
        # )

        self.classifier = nn.Linear(self.emb_dim, self.hparams.num_classes)

        # Metrics
        self.criterion = nn.CrossEntropyLoss()
# -----------------------------
        # ✅ Metrics (VAL)
        # -----------------------------
        nc = self.hparams.num_classes
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='micro')
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        self.val_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='weighted')
        self.val_prec_macro = torchmetrics.Precision(task="multiclass", num_classes=nc, average='macro')
        self.val_recall_macro = torchmetrics.Recall(task="multiclass", num_classes=nc, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=nc, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=nc, average='macro')
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

        # -----------------------------
        # ✅ Metrics (TEST)
        # -----------------------------
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.test_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='micro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        self.test_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='weighted')
        self.test_prec_macro = torchmetrics.Precision(task="multiclass", num_classes=nc, average='macro')
        self.test_recall_macro = torchmetrics.Recall(task="multiclass", num_classes=nc, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=nc, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=nc, average='macro')
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

        self.val_preds_raw = []
        self.val_labels_raw = []
        self.test_preds_raw = []
        self.test_labels_raw = []
        

    def _encode_windows(self, windows_b_sct):
        B, S, C, T = windows_b_sct.shape
        # print("shape: ", B, S, C, T)x
        flat = windows_b_sct.reshape(B * S, C, T)
        # ✅ Conv1d expects [B, C, T]. If input is [B, T, C], fix it.
        # if flat.shape[1] < flat.shape[2]:
        #     flat = flat.permute(0, 2, 1)

        if self.hparams.model_name in ["method", "comodo", "primus", "imu2clip"]:
            sensor_encoder = self.backbone.sensor_model

        if self.hparams.model_name == "method":
            reps = sensor_encoder(flat)
        elif self.hparams.model_name == "imu2clip":
            flat_padded = self.backbone.sensor_padding(flat)
            reps = sensor_encoder(flat_padded)
        elif self.hparams.model_name == "primus":
            reps = sensor_encoder(flat)['mmcl']
        elif self.hparams.model_name == "mae":
            reps = self.backbone.model.forward_sensor_only(flat)
        elif self.hparams.model_name == "comodo":
            reps = sensor_encoder(flat)
        else:
            raise ValueError(f"Unknown model_name: {self.hparams.model_name}")

        reps = reps.reshape(B, S, -1)
        return reps
        
    def forward(self, windows_b_sct, lengths_b):
        seq_emb = self._encode_windows(windows_b_sct)  # [B, S, D]

        # 🔹 probe_mode별 처리
        if self.probe_mode == "meanpool":
            pooled = seq_emb.mean(dim=1)
            logits = self.classifier(pooled)
            return logits
        else:  # lstm
            packed = nn.utils.rnn.pack_padded_sequence(
                seq_emb, lengths_b.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.temporal_model(packed)
            h_last = h_n[-1]
            logits = self.classifier(h_last)
            return logits

    def _shared_step(self, batch):
        windows, lengths, targets, metas = batch
        windows = windows.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device)
        targets = targets.to(self.device)

        logits = self(windows, lengths)
        loss = self.criterion(logits, targets)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)  # AUROC/AP 용
        return loss, preds, probs, targets

    # -----------------------------
    # Train
    # -----------------------------
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # -----------------------------
    # Validation
    # -----------------------------
    def on_validation_epoch_start(self):
        self.val_preds_raw.clear()
        self.val_labels_raw.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, targets = self._shared_step(batch)
        # print("target ", targets)
        # 클래스 인덱스 기반 메트릭
        self.val_acc.update(preds, targets)
        self.val_f1_micro.update(preds, targets)
        self.val_f1_macro.update(preds, targets)
        self.val_f1_weighted.update(preds, targets)
        self.val_prec_macro.update(preds, targets)
        self.val_recall_macro.update(preds, targets)
        self.val_conf_matrix.update(preds, targets)

        # 확률 기반 메트릭
        self.val_auroc.update(probs, targets)
        self.val_ap.update(probs, targets)

        # 로깅/원시 저장
        self.val_preds_raw.append(preds.detach().cpu())
        self.val_labels_raw.append(targets.detach().cpu())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # print("Sample preds:", preds[:10].tolist())
        # print("Sample labels:", targets[:10].tolist())

    def on_validation_epoch_end(self):

        # scalar compute
        log_dict = {
            "val/acc": self.val_acc.compute(),
            "val/f1_micro": self.val_f1_micro.compute(),
            "val/f1_macro": self.val_f1_macro.compute(),
            "val/f1_weighted": self.val_f1_weighted.compute(),
            "val/precision_macro": self.val_prec_macro.compute(),
            "val/recall_macro": self.val_recall_macro.compute(),
            "val/auroc_macro": torch.nan_to_num(self.val_auroc.compute()),
            "val/ap_macro": torch.nan_to_num(self.val_ap.compute()),
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        # Confusion Matrix 그림
        cm_t = self.val_conf_matrix.compute().cpu().numpy()  # [C, C]
        fig, ax = plt.subplots(figsize=(8, 6))
        class_names = self.class_names or [str(i) for i in range(cm_t.shape[0])]
        sns.heatmap(cm_t, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title("Validation Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        if self.logger:
            self.logger.experiment.log({"val/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        # reset
        for m in [
            self.val_acc, self.val_f1_micro, self.val_f1_macro, self.val_f1_weighted,
            self.val_prec_macro, self.val_recall_macro, self.val_auroc, self.val_ap,
            self.val_conf_matrix
        ]:
            m.reset()

    # -----------------------------
    # Test
    # -----------------------------
    def on_test_epoch_start(self):
        self.test_preds_raw.clear()
        self.test_labels_raw.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, probs, targets = self._shared_step(batch)

        # 클래스 인덱스 기반
        self.test_acc.update(preds, targets)
        self.test_f1_micro.update(preds, targets)
        self.test_f1_macro.update(preds, targets)
        self.test_f1_weighted.update(preds, targets)
        self.test_prec_macro.update(preds, targets)
        self.test_recall_macro.update(preds, targets)
        self.test_conf_matrix.update(preds, targets)

        # 확률 기반
        self.test_auroc.update(probs, targets)
        self.test_ap.update(probs, targets)

        self.test_preds_raw.append(preds.detach().cpu())
        self.test_labels_raw.append(targets.detach().cpu())

        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        log_dict = {
            "test/acc": self.test_acc.compute(),
            "test/f1_micro": self.test_f1_micro.compute(),
            "test/f1_macro": self.test_f1_macro.compute(),
            "test/f1_weighted": self.test_f1_weighted.compute(),
            "test/precision_macro": self.test_prec_macro.compute(),
            "test/recall_macro": self.test_recall_macro.compute(),
            "test/auroc_macro": torch.nan_to_num(self.test_auroc.compute()),
            "test/ap_macro": torch.nan_to_num(self.test_ap.compute()),
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        cm_t = self.test_conf_matrix.compute().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        class_names = self.class_names or [str(i) for i in range(cm_t.shape[0])]
        sns.heatmap(cm_t, annot=True, fmt="d", cmap="Greens",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title("Test Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        if self.logger:
            self.logger.experiment.log({"test/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        for m in [
            self.test_acc, self.test_f1_micro, self.test_f1_macro, self.test_f1_weighted,
            self.test_prec_macro, self.test_recall_macro, self.test_auroc, self.test_ap,
            self.test_conf_matrix
        ]:
            m.reset()

    def configure_optimizers(self):
        # probe_mode별 optimizer 설정
        if self.probe_mode == "meanpool":
            params = list(self.classifier.parameters())
        else:
            params = list(self.temporal_model.parameters()) + list(self.classifier.parameters())
        return torch.optim.Adam(params, lr=self.hparams.lr)


def main(args):
    set_random_seed(42)
    args = set_module_params(args)

    # HWU-USP 분기
    if args.dataset_name == "HWU-USP":
        args.num_classes = 5
        args.threshold_epoch = 100
        datamodule = MethodDataModule(args, stage="linear_probe_lstm")
        datamodule.setup("fit")  # ✅ 추가
        backbone = get_backbone_with_mode(args)
        model = LinearProbeLSTM(args, backbone)

        # ✅ class_to_idx를 datamodule에서 그대로 가져와 class_names에 저장
         # class_names 자동 추출
        class_names = list(datamodule.train_dataset.class_to_idx.keys())
        print(f"✅ Loaded class names: {class_names}")
        model.class_names = class_names
        monitor = 'val/acc'

    else:
        args.num_classes = 14
        args.threshold_epoch = 100 # need only sensor data
        datamodule = MethodDataModule(args, stage='linear_probe')
        
        backbone = get_backbone_with_mode(args)
        model = LinearProbeLightningModule(args, backbone)
        monitor = 'val_acc'
    
    # 분산 환경이 초기화되었는지 확인
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 4  # 초기화되지 않았으면 (즉, 단일 프로세스 실행) 1로 설정
    sensor_tag = "_supervision" if args.supervision else ""
    run_name = f"{args.model_name}_{args.dataset_name}_{args.ckpt_name}_{backbone.hparams.batch_size*4}_{backbone.hparams.epochs}_inear_probe_{args.batch_size* world_size}_linearEpoch={args.linear_epochs}{sensor_tag}"
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logger = WandbLogger(project="Method_Linear_Probe", name=run_name) if is_master_process else False
    # ckpt_cb = ModelCheckpoint(monitor=monitor, mode='max',
    #                           dirpath=f'checkpoints_linear/{args.model_name}_{args.dataset_name}',
    #                           filename='best-{epoch:02d}-{val_acc:.3f}', save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=args.linear_epochs,
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,
        # callbacks=[ckpt_cb]
    )

    print("--- Starting Linear Probing ---")
    trainer.fit(model, datamodule)
    datamodule.setup("fit")

    print("--- Testing ---")
    trainer.test(model=model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--linear_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--probe_mode', type=str, default='meanpool', choices=['lstm', 'meanpool'], help="Choose probing strategy: 'lstm' for temporal model or 'meanpool' for simple temporal average pooling")
    parser.add_argument('--supervision', type=bool, default=False, help='If True, skip backbone weight loading and run with frozen hyperparameter configs only')
    args = parser.parse_args()
    main(args)