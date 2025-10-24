import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from .model import CAVMAE
from baseline_modules import BasePretrainModule


class CAVMAELightningModule(BasePretrainModule):
    def __init__(self, args):
        super().__init__(args)
        
        # CAVMAE 모델 인스턴스 생성
        # DataModule에서 채널 수 등을 args에 추가해주면 더 좋습니다.
        self.model = CAVMAE(
            sensor_in_chans=self.hparams.num_sensors, # 예시: 36
            # embed_dim=self.hparams.embedding_dim,
            sensor_seq_len=self.hparams.sensor_seq_len, # 예시: 128
            norm_pix_loss=self.hparams.norm_pix_loss
        )

    def training_step(self, batch, batch_idx):
        video, sensor, _, _ = batch

        # CAVMAE 모델의 forward 호출
        loss, loss_mae, loss_mae_s, loss_mae_v, loss_c, c_acc = self.model(
            sensor=sensor,
            video=video,
            mask_ratio_s=self.hparams.masking_ratio,
            mask_ratio_v=self.hparams.masking_ratio,
            mae_loss_weight=self.hparams.mae_loss_weight,
            contrast_loss_weight=self.hparams.contrast_loss_weight,
        )

        # ⭐️ pretrain_loss로 로그를 남겨야 ModelCheckpoint가 모니터링 가능
        self.log("pretrain_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("mae_loss", loss_mae, logger=True, on_step=True, on_epoch=True)
        self.log("sensor_mae_loss", loss_mae_s, logger=True, on_step=True, on_epoch=True)
        self.log("video_mae_loss", loss_mae_v, logger=True, on_step=True, on_epoch=True)
        self.log("contrastive_loss", loss_c, logger=True, on_step=True, on_epoch=True)
        self.log("contrastive_acc", c_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        # 원본 코드의 옵티마이저와 스케줄러 설정
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=5e-7, 
            betas=(0.95, 0.999)
        )
        
        scheduler = MultiStepLR(
            optimizer, 
            milestones=list(range(self.hparams.lrscheduler_start, 1000, self.hparams.lrscheduler_step)),
            gamma=self.hparams.lrscheduler_decay
        )
        
        return [optimizer], [scheduler]