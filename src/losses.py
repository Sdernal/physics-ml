import torch
import torch.nn as nn


class DirichletLoss(nn.Module):
    def __init__(self, left_p, right_p):
        super().__init__()
        self.left_p = left_p
        self.right_p = right_p

    def forward(self, p):
        # p - batch_size, 64, 64
        # dp / dy = 0 at y = 0 and y = y_max
        bottom_loss = torch.mean((p[:, :, 1] - p[:, :, 0]) ** 2)
        top_loss = torch.mean((p[:, :, -2] - p[:, :, -1]) ** 2)
        # also p should satisfy left and right conditions
        left_loss = torch.mean((p[:, 0, :] - self.left_p) ** 2)
        right_loss = torch.mean((p[:, - 1, :] - self.right_p) ** 2)
        return (bottom_loss + top_loss + left_loss + right_loss) / 4


class LaplacianLoss(nn.Module):

    def __init__(self, dx):
        super().__init__()
        self.dx = dx
        self.mse = nn.MSELoss()

    def forward(self, p, g):
        laplacian_p = ((p[:, 2:, 1:-1] - p[:, 1:-1, 1:-1] * 2 + p[:, :-2, 1:-1]) / self.dx ** 2
                       + (p[:, 1:-1, 2:] - p[:, 1:-1, 1:-1] * 2 + p[:, 1:-1, :-2]) / self.dx ** 2)
        return self.mse(laplacian_p, g) * p.shape[-1] ** 2 * p.shape[-2] ** 2  # multiply by Lx^2 * Ly^2 as in paper
