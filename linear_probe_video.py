import os
import io
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pytorchvideo.models.hub import x3d_s  # ✅ X3D backbone
from datamodule import MethodDataModule


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DatasetEpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(trainer.datamodule, 'train_dataset') and hasattr(trainer.datamodule.train_dataset, 'set_epoch'):
            trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch)
        if hasattr(trainer.datamodule, 'val_dataset') and hasattr(trainer.datamodule.val_dataset, 'set_epoch'):
            trainer.datamodule.val_dataset.set_epoch(trainer.current_epoch)


####################################################################
class VideoLinearProbeModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        print("🧠 Loading X3D-S backbone (Kinetics pretrained)...")
        self.backbone = x3d_s(pretrained=True)

        # X3D 기본 classifier 제거
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Identity()

        # freeze or unfreeze
        if self.hparams.unfreeze_backbone:
            print("🟢 Fine-tuning: X3D backbone unfrozen.")
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            print("🔒 Linear probe: X3D backbone frozen.")
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # linear probe
        self.probe = nn.Linear(in_features, self.hparams.num_classes)

        # metrics
        task_type = "multiclass"
        self.train_accuracy = torchmetrics.Accuracy(task=task_type, num_classes=self.hparams.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task=task_type, num_classes=self.hparams.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task=task_type, num_classes=self.hparams.num_classes)
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task=task_type, num_classes=self.hparams.num_classes)
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(task=task_type, num_classes=self.hparams.num_classes)

    def forward(self, video_batch):
        # [B, T, C, H, W] → pytorchvideo expects [B, C, T, H, W]
        video_batch = video_batch.permute(0, 2, 1, 3, 4)
        with torch.no_grad() if not self.hparams.unfreeze_backbone else torch.enable_grad():
            features = self.backbone(video_batch)
        logits = self.probe(features)
        return logits

    def _shared_step(self, batch):
        videos, sensors, labels, _ = batch
        logits = self(videos)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.train_accuracy.update(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.val_accuracy.update(preds, labels)
        self.val_conf_matrix.update(preds, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.test_accuracy.update(preds, labels)
        self.test_conf_matrix.update(preds, labels)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        try:
            cm = self.val_conf_matrix.compute().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=self.hparams.class_names,
                        yticklabels=self.hparams.class_names, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Validation Confusion Matrix (epoch {self.current_epoch})")

            save_dir = "/home/jaemo/Method/wandbs"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch_{self.current_epoch:03d}_confusion_matrix.png")
            plt.savefig(save_path, bbox_inches="tight")

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            run = wandb.run or getattr(self.logger, "experiment", None)
            if run:
                run.log({"val_confusion_matrix": wandb.Image(img),
                         "trainer/epoch": self.current_epoch})
            plt.close(fig)
        except Exception as e:
            print(f"Failed to log validation confusion matrix: {e}")
        self.val_conf_matrix.reset()

    def on_test_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        try:
            cm = self.test_conf_matrix.compute().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                        xticklabels=self.hparams.class_names,
                        yticklabels=self.hparams.class_names, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Test Confusion Matrix")
            save_dir = "/home/jaemo/Method/wandbs"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch_{self.current_epoch:03d}_test_confusion_matrix.png")
            plt.savefig(save_path, bbox_inches="tight")

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            run = wandb.run or getattr(self.logger, "experiment", None)
            if run:
                run.log({"test_confusion_matrix": wandb.Image(img),
                         "trainer/epoch": self.current_epoch})
            plt.close(fig)
        except Exception as e:
            print(f"Failed to log test confusion matrix: {e}")
        self.test_conf_matrix.reset()

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.probe.parameters()) \
            if self.hparams.unfreeze_backbone else self.probe.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


####################################################################
def main(args):
    set_random_seed(args.seed)
    datamodule = MethodDataModule(args, args.stage)

    ACTION_MERGE_LABELS_OPPORTUNITY = {
        0: 'Open Door 1', 1: 'Open Door 2', 2: 'Close Door 1', 3: 'Close Door 2',
        4: 'Open Fridge', 5: 'Close Fridge', 6: 'Open Dishwasher', 7: 'Close Dishwasher',
        8: 'Open Drawer 1', 9: 'Close Drawer 1', 10: 'Open Drawer 2', 11: 'Close Drawer 2',
        12: 'Open Drawer 3', 13: 'Close Drawer 3'
    }

    args.num_classes = 14
    args.class_names = [ACTION_MERGE_LABELS_OPPORTUNITY[i] for i in range(args.num_classes)]

    model = VideoLinearProbeModel(args)

    logging_name = f"{'finetune' if args.unfreeze_backbone else 'linear_probe'}_{args.dataset_name}_X3D"
    logger = WandbLogger(project=args.project_name, name=logging_name,
                         save_dir="/home/jaemo/Method/wandbs", log_model=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints_x3d/{args.dataset_name}",
        filename="best_model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1, monitor="val_acc", mode="max", save_last=True
    )
    callbacks = [DatasetEpochCallback(), checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator='gpu', devices=args.devices,
        strategy='ddp_find_unused_parameters_true', logger=logger, callbacks=callbacks
    )

    print("--- Starting Training ---")
    trainer.fit(model, datamodule)
    print("\n--- Testing Best Model ---")
    trainer.test(datamodule=datamodule, ckpt_path='best')

    if args.unfreeze_backbone:
        save_path = f"./checkpoints_x3d/{args.dataset_name}/x3d_finetuned_epoch{args.epochs}.pth"
        torch.save(model.backbone.state_dict(), save_path)
        print(f"✅ Saved fine-tuned X3D backbone to {save_path}")


####################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Linear Probe / Fine-tuning with X3D")
    parser.add_argument("--dataset_name", type=str, default="Opportunity++")
    parser.add_argument("--project_name", type=str, default="Method_Linear_Probe")
    parser.add_argument("--model_name", type=str, default="video_linear")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_epoch", type=int, default=-3)
    parser.add_argument("--stage", type=str, default="pretrain")
    parser.add_argument("--unfreeze_backbone", action="store_true",
                        help="If set, unfreezes X3D backbone for fine-tuning and saves it after training.")
    args = parser.parse_args()
    main(args)
