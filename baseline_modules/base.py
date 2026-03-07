import pytorch_lightning as pl
import torch

class BasePretrainModule(pl.LightningModule):
    """
    모든 Pre-training 모듈이 상속받을 기본 클래스.
    공통 로직(예: 옵티마이저)을 여기에 정의할 수 있습니다.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.lr)
        return optimizer