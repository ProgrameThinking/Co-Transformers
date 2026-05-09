import torch.nn as nn
import torch.nn.functional as F

from common.utils import SobelEdgeExtractor
from models.segformer import SegFormerBackbone


class HierarchicalEdgeSupervisedTransformer(nn.Module):
    def __init__(self, pretrain_path='pretrained/mit_b3.pth', nclass=1):
        super().__init__()
        self.backbone = SegFormerBackbone(pretrain_path=pretrain_path)
        self.edge_supervised_module = EdgeSupervisedModule(nclass=nclass)

    def forward(self, x):
        _, features = self.backbone(x)
        F1, F2, F3, F_edge = features
        X_edge = self.edge_supervised_module(F1, F2, F3, F_edge)
        return F_edge, X_edge


class EdgeSupervisedModule(nn.Module):
    def __init__(self, nclass=1):
        super().__init__()
        feature_channels = (64, 128, 320, 512)
        self.edge_extractors = nn.ModuleList([
            SobelEdgeExtractor(channels, out_channels=1)
            for channels in feature_channels
        ])
        self.edge_projectors = nn.ModuleList([
            EdgeRefinementBlock(channels, nclass)
            for channels in feature_channels
        ])
        self.edge_mergers = nn.ModuleList([
            EdgeRefinementBlock(nclass, nclass)
            for _ in range(3)
        ])

    def forward(self, F1, F2, F3, F_edge):
        edge_features = [
            projector(extractor(feature))
            for extractor, projector, feature in zip(
                self.edge_extractors,
                self.edge_projectors,
                (F1, F2, F3, F_edge),
            )
        ]

        merged_edge = edge_features[0]
        for level, scale_factor in enumerate((2, 4, 4), start=1):
            resized_edge = F.interpolate(
                edge_features[level],
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=True,
            )
            merged_edge = merged_edge + resized_edge
            merged_edge = self.edge_mergers[level - 1](
                merged_edge,
                relu=level != len(edge_features) - 1,
            )
        return merged_edge


class EdgeRefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res) if relu else x + res
