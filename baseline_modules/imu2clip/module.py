# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import torch
import pytorch_lightning as pl
import torchmetrics
from .model import MW2StackRNNPooling, ClipPLModel
from baseline_modules.base import BasePretrainModule
from baseline_modules.loss import InfoNCE

class IMU2CLIPLightningModule(BasePretrainModule):
    def __init__(self):
        super().__init__()

        self.loss = InfoNCE(symmetric_loss=True, learn_temperature=True)

        self.sensor_model = MW2StackRNNPooling(size_embeddings=self.hparams.embedding_dim)
        self.video_model = ClipPLModel(freeze=True)

    def forward(self, batch):
        # x_sensor: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings

        out = {}

        videos, sensors, labels, sample_ids = batch
        x_sensor = sensors
        y_sensor = self.sensor_model(x_sensor)
        out["sensor"] = y_sensor

        x_video = videos
        y_video = self.video_model.get_video_embeddings(x_video)
        out["video"] = y_video

        return out

    def training_step(self, batch, batch_idx: int):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch)

        # Use NCE loss
        y_query_modality = y["sensor"]
        loss_output = 0.0

        # Compute loss for source modality <> each target modality
        y_key_modality = y["video"]
        s2t_loss = self.loss(query=y_query_modality, positive_key=y_key_modality)

        # Log the loss
        str_s2t = "i2v"
        self.log(f"train_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
        loss_output += s2t_loss

        self.log("train_loss", loss_output, logger=True, sync_dist=True)
        return loss_output

    def predict_step(self, batch, batch_idx: int):
        return self(batch)
