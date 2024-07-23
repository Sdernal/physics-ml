from pathlib import Path
from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset


class PoissonDataset(Dataset):
    def __init__(self, data_path: Union[Path, str], dt: float = 0.001, h: float = 1., rho: float = 1.):
        data_archive = np.load(data_path)
        assert len({'p', 'u', 'v'} & set(data_archive.keys())) == 3

        self.pt = data_archive['p']
        vt = data_archive['v']
        ut = data_archive['u']
        dx = dy = h

        self.g = np.zeros_like(vt)
        self.g[:, 1:-1, 1:-1] = ((ut[:, 2:, 1:-1] - ut[:, :-2, 1:-1]) / (2 * dx)
                                 + (vt[:, 1:-1, 2:] - vt[:, 1:-1, :-2]) / (2 * dy)) * rho / dt

        g2 = np.zeros_like(self.g)
        g2[:, 1:-1, 1:-1] = (
                (ut[:, 2:, 1:-1] - ut[:, :-2, 1:-1]) ** 2 / (4 * dx ** 2)
                + (vt[:, 1:-1, 2:] - vt[:, 1:-1, :-2]) ** 2 / (4 * dy ** 2)
                + (ut[:, 1:-1, 2:] - ut[:, 1:-1, :-2]) * (vt[:, 2:, 1:-1] - vt[:, :-2, 1:-1]) / (2 * dx * dy)
        ) * rho

        self.g -= g2
        # TODO: add normalization and maybe border values

    def __len__(self):
        return self.pt.shape[0] - 2

    def __getitem__(self, item):
        return torch.Tensor(self.g[item + 1]), torch.Tensor(self.pt[item + 2])

    @staticmethod
    def collate_fn(batch):
        g, p = list(zip(*batch))
        return torch.stack(g, dim=0), torch.stack(p, dim=0)
