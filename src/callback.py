import numpy as np
import torch
from typing import Any, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger


class LogPressureCallback(Callback):
    def __init__(self, logger: WandbLogger, indices: List[int] = None, dx: float = 1., p_max=None, g_max=None):
        if indices is None:
            self.indices = [0, -1]  # draw first and last frame
        else:
            self.indices = indices
        self.logger = logger
        self.dx = dx
        self.gt = []
        self.pt = []
        self.p_pred_t = []
        self.p_max = p_max
        self.g_max = g_max

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # save inputs and outputs to show later
        g, p = batch
        p_pred = outputs
        self.gt.append(g.cpu().numpy())
        self.pt.append(p.cpu().numpy())
        self.p_pred_t.append(p_pred.cpu().numpy())

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # take first and last predictions

        gt_cat = np.concatenate(self.gt, axis=0)
        pt_cat = np.concatenate(self.pt, axis=0)
        p_pred_t_cat = np.concatenate(self.p_pred_t, axis=0)

        def prepare_frame(idx):
            if idx < 0:
                idx = gt_cat.shape[0] + idx
            if idx > gt_cat.shape[0]:
                idx = idx % gt_cat.shape[0]

            g, p, p_pred = gt_cat[idx], pt_cat[idx], p_pred_t_cat[idx]
            laplacian = np.zeros_like(g)
            p_denorm = p_pred * self.p_max if self.p_max is not None else p_pred
            laplacian[1:-1, 1:-1] = (
                        (p_denorm[2:, 1:-1] - p_denorm[1:-1, 1:-1] * 2 + p_denorm[:-2, 1:-1]) / self.dx ** 2
                        + (p_denorm[1:-1, 2:] - p_denorm[1:-1, 1:-1] * 2 + p_denorm[1:-1, :-2]) / self.dx ** 2)

            images_tmp = [np.rot90(p_pred), np.rot90(p), np.rot90(laplacian),
                          np.rot90(g*self.g_max if self.g_max is not None else g)]
            captions_tmp = [f'p_pred_{idx}', f'p_true_{idx}', f'laplacian_{idx}', f'g_{idx}']
            return images_tmp, captions_tmp

        images = []
        captions = []
        for idx in self.indices:
            i, c = prepare_frame(idx)
            images.extend(i)
            captions.extend(c)
        self.logger.log_image(key="sample_images", images=images, caption=captions)

        self.gt = []
        self.pt = []
        self.p_pred_t = []
