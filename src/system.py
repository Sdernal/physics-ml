import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .model import UNet
from .dataset import PoissonDataset
from .losses  import DirichletLoss, LaplacianLoss


class PoissonSolver(pl.LightningModule):
    def __init__(self, trn_path, val_path, dt: float = 0.001, batch_size: int = 32,
                 p_max: float = None, g_max: float = None):
        super().__init__()
        # TODO: move hardcoded params
        self.trn_dataset = PoissonDataset(trn_path, dt=dt, p_max=p_max, g_max=g_max)
        self.val_dataset = PoissonDataset(val_path, dt=dt, p_max=p_max, g_max=g_max)
        self.model = UNet()
        self.dirichlet_loss = DirichletLoss(left_p=5 / p_max, right_p=2. / p_max)
        self.inside_loss = nn.MSELoss()
        self.laplacian_loss = LaplacianLoss(dx=1.)
        self.batch_size = batch_size
        self.p_max = p_max
        self.g_max = g_max

    def forward(self, g):
        p_pred = self.model(g)
        return p_pred

    def training_step(self, batch, batch_idx):
        g, p = batch
        p_pred = self.forward(g)
        loss = self.custom_loss(g, p, p_pred, 'trn')
        return loss

    def validation_step(self, batch, batch_idx):
        g, p = batch
        p_pred = self.forward(g)
        loss = self.custom_loss(g, p, p_pred, 'val')
        return p_pred

    def custom_loss(self, g, p, p_pred, log_prefix):
        dirichlet_loss = self.dirichlet_loss(p_pred)
        inside_loss = self.inside_loss(p_pred[:, 1:-1, 1:-1], p[:, 1:-1, 1:-1])
        laplacian_loss = self.laplacian_loss(p_pred * self.p_max, g * self.g_max)  # denormalize
        loss = dirichlet_loss + inside_loss + laplacian_loss
        # TODO: add coefficients and think about denormalization on laplacian loss

        # Log losses
        self.log(f'{log_prefix}/loss', loss)
        self.log(f'{log_prefix}/dirichlet_loss', dirichlet_loss)
        self.log(f'{log_prefix}/inside_loss', inside_loss)
        self.log(f'{log_prefix}/laplacian_loss', laplacian_loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())  # TODO: look for parameters in paper

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.trn_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn)