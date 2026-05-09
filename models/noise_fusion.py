import torch
import torch.nn as nn

from common.noiseprint import make_net
from common.layers import SRMFilter, BayarConv2d


class NoiseFusionModule(nn.Module):
    def __init__(self, noiseprint_path='pretrained/np++.pth'):
        super().__init__()

        num_levels = 17
        self.noiseprint = make_net(
            3,
            kernels=[3] * num_levels,
            features=[64] * (num_levels - 1) + [1],
            bns=[False] + [True] * (num_levels - 2) + [False],
            acts=['relu'] * (num_levels - 1) + ['linear'],
            dilats=[1] * num_levels,
            bn_momentum=0.1,
            padding=1,
        )

        if noiseprint_path:
            weights = torch.load(noiseprint_path, map_location=torch.device('cpu'))
            self.noiseprint.load_state_dict(weights)

        self.noiseprint.eval()
        for param in self.noiseprint.parameters():
            param.requires_grad = False

        self.bayar = BayarConv2d(3, 3, padding=2)
        self.srm = SRMFilter()
        self.feature_fusion = NoiseFeatureFusion(num_inputs=3)

    def forward(self, x):
        noiseprint = self.noiseprint(x)
        if noiseprint.size(1) == 1:
            noiseprint = noiseprint.repeat(1, 3, 1, 1)

        return self.feature_fusion([
            noiseprint,
            self.bayar(x),
            self.srm(x),
        ])


class NoiseFeatureFusion(nn.Module):
    def __init__(self, num_inputs=3, out_channels=3):
        super().__init__()
        self.input_adapters = nn.ModuleList([
            EarlyConv()
            for _ in range(num_inputs)
        ])
        self.dropout = nn.Dropout(0.33)
        self.output_adapter = EarlyConv(in_channels=3 * num_inputs, out_channels=out_channels)

    def forward(self, noise_views):
        features = [
            adapter(noise_view)
            for adapter, noise_view in zip(self.input_adapters, noise_views)
        ]
        return self.output_adapter(self.dropout(torch.cat(features, dim=1)))


class EarlyConv(nn.Module):
    def __init__(self, depth=3, in_channels=3, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        channel_plan = [in_channels] + [24 * 2 ** i for i in range(depth)]
        self.body = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channel_plan[i], channel_plan[i + 1], 3, 1, 'same'),
                nn.BatchNorm2d(channel_plan[i + 1]),
                nn.ReLU(),
            )
            for i in range(depth)
        ])
        self.proj = nn.Conv2d(channel_plan[-1], out_channels, 1, 1, 'same')

    def forward(self, x):
        return self.proj(self.body(x))
