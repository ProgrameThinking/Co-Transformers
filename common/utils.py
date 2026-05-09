import torch
import torch.nn as nn


class SobelEdgeExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        branches = []
        for kernel in (
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ],
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1],
            ],
        ):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            weight = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 3, 3)
            conv.weight = nn.Parameter(weight.repeat(out_channels, in_channels, 1, 1), requires_grad=False)
            branches.append(nn.Sequential(conv, nn.BatchNorm2d(out_channels)))

        self.gradient_x, self.gradient_y = branches

    def forward(self, x):
        gradient = torch.sqrt(torch.pow(self.gradient_x(x), 2) + torch.pow(self.gradient_y(x), 2))
        return torch.sigmoid(gradient) * x


def dice_loss(out, gt, smooth=1.0):
    gt = gt.view(-1)
    out = out.view(-1)
    intersection = (gt * out).sum()
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(out).sum() + smooth)
    return 1.0 - dice
