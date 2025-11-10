import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from torchmetrics import Accuracy
import os

# --- 사용자 정의 모듈 임포트 ---
# 이 클래스들은 별도의 파일에 정의되어 있다고 가정합니다.
from baseline_modules.loss import COMODOLoss
from .model import VideoTeacherMLP, IMUStudentMLP, create_pipeline
from baseline_modules.base import BasePretrainModule

class COMODOLightningModule(BasePretrainModule):
    def __init__(self, args, instance_queue_encoded=None, anchor_video_embeddings=None):
        super().__init__(args)
        # --- 1. 모델 정의 ---
        # Video Teacher (학습되지 않음)
        self.video_teacher = VideoTeacherMLP(
            self.hparams.video_ckpt, self.hparams.mlp_output_dim, self.hparams.mlp_hidden_dim, 'cuda'
        )
        self.video_teacher.eval()
        for param in self.video_teacher.parameters():
            param.requires_grad = False

        # IMU Student (학습 대상)
        imu_pipeline = create_pipeline(
            self.hparams.imu_ckpt, self.hparams.num_classes, 'cuda', self.hparams.reduction, self.hparams.num_sensors
        )
        self.sensor_model = IMUStudentMLP(
            imu_pipeline, 'cpu', self.hparams.mlp_output_dim, self.hparams.mlp_hidden_dim,
            activation_fn=nn.GELU, reduction=self.hparams.reduction, num_sensors=self.hparams.num_sensors
        )

        # --- 2. Loss 함수 정의 ---
        if instance_queue_encoded is not None:
            self.comodo_loss = COMODOLoss(
                instanceQ_encoded=instance_queue_encoded.cpu(), # device는 trainer가 관리
                student_model=self.sensor_model,
                teacher_temp=self.hparams.teacher_temp,
                student_temp=self.hparams.student_temp,
            )

        # --- 3. 평가용 데이터 및 메트릭 ---
        # register_buffer: device 이동(예: .to('cuda'))을 자동으로 처리
        self.register_buffer("anchor_video_embeddings", anchor_video_embeddings)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, imu_data, input_mask=None):
        # forward는 학생 모델의 추론을 담당
        return self.sensor_model(imu_data, input_mask)

    def training_step(self, batch, batch_idx):
        videos, sensors, labels, sample_ids, _ = batch
        idx, video_id = sample_ids
        input_mask = None # None when Mantis
         # 4. 캐시 확인 및 로드
        encoded_video_list = []
        for vid, video_tensor in zip(video_id, videos):
            cache_path = os.path.join(self.hparams.baseline_video_cache_dir, f"{vid}.pt")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            if os.path.exists(cache_path):
                encoded_video = torch.load(cache_path, map_location=self.device)
            else:
                encoded_video = self.video_teacher.encode(video_tensor.unsqueeze(0))
                torch.save(encoded_video, cache_path)
                print(f"Cached feature for {vid}")
            if encoded_video.dim() == 1:
                encoded_video = encoded_video.unsqueeze(0)
            encoded_video_list.append(encoded_video)
        z_v = torch.cat(encoded_video_list, dim=0)
        loss = self.comodo_loss(sensors, z_v, input_mask=input_mask)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # --- 옵티마이저 및 스케줄러 설정 ---
        optimizer = torch.optim.AdamW(self.sensor_model.parameters(), lr=self.hparams.learning_rate)
        
        # 스케줄러 계산
        num_training_steps = len(self.trainer.datamodule.train_dataloader()) * self.hparams.epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # 매 스텝마다 스케줄러 호출
            },
        }
    