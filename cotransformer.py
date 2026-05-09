import torch
import torch.nn as nn
import torch.nn.functional as F

from IMDLBenCo.registry import MODELS
from common.utils import dice_loss
from models.hes_transformer import HierarchicalEdgeSupervisedTransformer
from models.noise_fusion import NoiseFusionModule
from models.cte_transformer import CrossTraceExtractionTransformer
from common.pred_head import PredictionModule


@MODELS.register_module()
class CoTransformers(nn.Module):
    def __init__(
        self,
        nclass=1,
        segformer_pretrain_path='pretrained/mit_b3.pth',
        noiseprint_path='pretrained/noiseprint.pth',
    ):
        super().__init__()
        self.num_class = nclass
        self.hierarchical_edge_supervised_transformer = HierarchicalEdgeSupervisedTransformer(
            pretrain_path=segformer_pretrain_path,
            nclass=nclass,
        )
        self.noise_fusion_module = NoiseFusionModule(noiseprint_path=noiseprint_path)
        self.cross_trace_extraction_transformer = CrossTraceExtractionTransformer()
        self.prediction_module = PredictionModule(512 + 768, nclass)

    def forward(self, image, mask, edge_mask, **kwargs):
        target_size = image.shape[-2:]

        edge_feature, edge_logits = self.hierarchical_edge_supervised_transformer(image)
        noise_inputs = self.noise_fusion_module(image)
        noise_feature = self.cross_trace_extraction_transformer(noise_inputs)

        fused_feature = torch.cat((edge_feature, noise_feature), dim=1)
        mask_logits = self.prediction_module(fused_feature)[0]
        mask_logits = F.interpolate(mask_logits, target_size, mode='bilinear', align_corners=True)
        edge_logits = F.interpolate(edge_logits, target_size, mode='bilinear', align_corners=True)

        out_mask = torch.sigmoid(mask_logits)
        out_edge = torch.sigmoid(edge_logits)
        loss_pixel = dice_loss(out_mask, mask)
        loss_edge = dice_loss(out_edge, edge_mask)
        loss = 0.8 * loss_edge + 0.16 * loss_pixel

        return {
            'backward_loss': loss,
            'pred_mask': out_mask,
            'visual_loss': {
                'pixel_loss': loss_pixel,
                'edge_loss': loss_edge,
                'total_loss': loss,
            },
            'visual_image': {
                'pred_mask': out_mask,
                'edge_mask': edge_mask,
            },
        }
