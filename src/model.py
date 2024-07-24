import torch
import torch.nn as nn
import torch.nn.functional as F


class Up(nn.Module):
    def __init__(self, c1: int, c2: int, c_out: int, factor: float = 2):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=factor, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(c1+c2, c_out, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode='zeros'),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 - batch_size, c1, h, w
        # x2 - batch_size, c2, 2*h, 2*w
        x1 = self.up(x1)  # batch_size, c1, 2*h, 2*w
        x = torch.cat([x1, x2], dim=1)  # batch_size, c1 + c2, 2*h, 2*w
        x = self.conv(x)  # batch_size, c_out, 2*h, 2*w
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros')
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros'),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros')
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(48, 62, 3, padding=1, padding_mode='zeros')
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(62, 60, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(60, 62, 3, padding=1, padding_mode='zeros')
        )

        self.up1 = Up(62, 62, 64)
        self.up2 = Up(64, 32, 64)
        self.up3 = Up(64, 32, 32)
        self.up4 = Up(32, 32, 32)

        self.out = nn.Sequential(
            # nn.Conv2d(32, 16, 3, padding=1, padding_mode='zeros'),
            # nn.Conv2d(16, 16, 3, padding=1, padding_mode='zeros'),
            nn.Conv2d(32, 1, 3, padding=1, padding_mode='zeros'),
        )

    def forward(self, x: torch.Tensor):
        # x - (batch_size, 64, 64)
        x = x.unsqueeze(1)  # batch_size, 1, 64, 64
        x1 = self.input(x)  # batch_size, 32, 64, 64
        x2 = self.down1(x1)  # batch_size, 32, 32, 32
        x3 = self.down2(x2)  # batch_size, 32, 16, 16
        x4 = self.down3(x3)  # batch_size, 62, 8, 8
        x = self.down4(x4)  # batch_size, 62, 4, 4

        x = self.up1(x, x4)  # batch_size, 64, 8, 8
        x = self.up2(x, x3)  # batch_size, 64, 16, 16
        x = self.up3(x, x2)  # batch_size, 32, 32, 32
        x = self.up4(x, x1)  # batch_size, 32, 64, 64
        x = self.out(x)  # batch_size, 1, 64, 64
        x = x.squeeze(dim=1)  # batch_size, 64, 64
        x = F.sigmoid(x)  # force to [0,1]
        return x
