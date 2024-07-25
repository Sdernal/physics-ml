from pathlib import Path
from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import calculate_g


class PoissonDataset(Dataset):
    def __init__(self, data_path: Union[Path, str], dt: float = 0.001, h: float = 1., rho: float = 1.,
                 p_max: float = None, g_max: float = None):
        data_archive = np.load(data_path)
        assert len({'p', 'u', 'v'} & set(data_archive.keys())) == 3

        self.pt = data_archive['p']
        vt = data_archive['v']
        ut = data_archive['u']
        self.g = calculate_g(ut, vt, rho, dt, h)

        # TODO: add normalization and maybe border values
        self.p_max = p_max
        self.g_max = g_max
        if self.p_max is not None:
            self.pt = self.pt / self.p_max
        if self.g_max is not None:
            self.g = self.g / self.g_max

    def __len__(self):
        return self.pt.shape[0] - 2

    def __getitem__(self, item):
        return torch.Tensor(self.g[item + 1]), torch.Tensor(self.pt[item + 2])

    @staticmethod
    def collate_fn(batch):
        g, p = list(zip(*batch))
        return torch.stack(g, dim=0), torch.stack(p, dim=0)
