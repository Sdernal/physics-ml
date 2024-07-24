import torch
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger


class LogPressureCallback(Callback):
    def __init__(self, logger: WandbLogger, dx: float = 1.):
        self.logger = logger
        self.dx = dx

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # log first image
        # logger = trainer.logger  # type: WandbLogger
        if batch_idx == 0:
            g, p = batch
            p_pred = outputs
            laplacian = torch.zeros_like(g)
            laplacian[:, 1:-1, 1:-1] = ((p_pred[:, 2:, 1:-1] - p_pred[:, 1:-1, 1:-1] * 2 + p_pred[:, :-2, 1:-1]) / self.dx ** 2
                       + (p_pred[:, 1:-1, 2:] - p_pred[:, 1:-1, 1:-1] * 2 + p_pred[:, 1:-1, :-2]) / self.dx ** 2)
            images = [p_pred[0], p[0],laplacian[0], g[0]]
            captions = ['p_pred', 'p_true', 'laplacian', 'g']
            self.logger.log_image(key="sample_images", images=images, caption=captions)