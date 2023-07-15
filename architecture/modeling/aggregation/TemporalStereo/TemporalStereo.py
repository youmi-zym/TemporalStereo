import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from detectron2.config import configurable

from ..builder import AGGREGATION_REGISTRY
from .coarse import CoarseAggregation
from .fine import FineAggregation
from .precise import PreciseAggregation


@AGGREGATION_REGISTRY.register()
class TEMPORALSTEREO(nn.Module):
    """
    Cost Aggregation method proposed by TemporalStereo
    Args:
        max_disp:                       (int), the max disparity value
        norm:                           (str), the type of normalization layer
        activation:                     (str, list, tuple), the type of activation layer and its coefficient is needed
    """
    @configurable
    def __init__(self,
                 coarse: nn.Module,
                 fine: nn.Module,
                 precise: nn.Module,
                 norm: str = 'BN',
                 activation: Union[str, List, Tuple] = 'SiLU'):
        super(TEMPORALSTEREO, self).__init__()
        self.norm = norm
        self.activation = activation

        self.coarse = coarse
        self.fine = fine
        self.precise = precise

    @classmethod
    def from_config(cls, cfg):
        coarse = CoarseAggregation(
            in_planes=          cfg.MODEL.AGGREGATION.COARSE.get('IN_PLANES', 192),
            C=                  cfg.MODEL.AGGREGATION.COARSE.get('C', 32),
            num_sample=         cfg.MODEL.AGGREGATION.COARSE.get('NUM_SAMPLE', 12),
            delta=              cfg.MODEL.AGGREGATION.COARSE.get('DELTA', 1),
            block_cost_scale=   cfg.MODEL.AGGREGATION.COARSE.get('BLOCK_COST_SCALE', 3),
            topk=               cfg.MODEL.AGGREGATION.COARSE.get('TOPK', 2),
            spatial_fusion=     cfg.MODEL.AGGREGATION.COARSE.get('SPATIAL_FUSION', True),
            norm=               cfg.MODEL.AGGREGATION.COARSE.get('NORM', 'BN3d'),
            activation=         cfg.MODEL.AGGREGATION.COARSE.get('ACTIVATION', 'SiLU'),
        )
        fine = FineAggregation(
            in_planes=          cfg.MODEL.AGGREGATION.FINE.get('IN_PLANES', 64),
            C=                  cfg.MODEL.AGGREGATION.FINE.get('C', 16),
            num_sample=         cfg.MODEL.AGGREGATION.FINE.get('NUM_SAMPLE', 5),
            delta=              cfg.MODEL.AGGREGATION.FINE.get('DELTA', 1),
            block_cost_scale=   cfg.MODEL.AGGREGATION.FINE.get('BLOCK_COST_SCALE', 3),
            topk=               cfg.MODEL.AGGREGATION.FINE.get('TOPK', 2),
            spatial_fusion=     cfg.MODEL.AGGREGATION.FINE.get('SPATIAL_FUSION', True),
            norm=               cfg.MODEL.AGGREGATION.FINE.get('NORM', 'BN3d'),
            activation=         cfg.MODEL.AGGREGATION.FINE.get('ACTIVATION', 'SiLU'),
        )
        precise = PreciseAggregation(
            in_planes=          cfg.MODEL.AGGREGATION.PRECISE.get('IN_PLANES', 48),
            C=                  cfg.MODEL.AGGREGATION.PRECISE.get('C', 8),
            num_sample=         cfg.MODEL.AGGREGATION.PRECISE.get('NUM_SAMPLE', 5),
            delta=              cfg.MODEL.AGGREGATION.PRECISE.get('DELTA', 1),
            block_cost_scale=   cfg.MODEL.AGGREGATION.PRECISE.get('BLOCK_COST_SCALE', 3),
            topk=               cfg.MODEL.AGGREGATION.PRECISE.get('TOPK', 2),
            norm=               cfg.MODEL.AGGREGATION.PRECISE.get('NORM', 'BN3d'),
            activation=         cfg.MODEL.AGGREGATION.PRECISE.get('ACTIVATION', 'SiLU'),
        )
        return {
            'coarse':                   coarse,
            'fine':                     fine,
            'precise':                  precise,
            "norm":                     cfg.MODEL.AGGREGATION.get('NORM', 'BN'),
            "activation":               cfg.MODEL.AGGREGATION.get('ACTIVATION', 'SiLU'),
        }

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left_feats, right_feats, left_image, right_image, prev_info: dict):
        disps = []
        costs = []
        offs = []
        disp_samples = []
        search_ranges = []
        disp_range = 4

        left_feat, left_feat_8, left_feat_16 = left_feats
        right_feat, right_feat_8, right_feat_16 = right_feats

        # coarse prediction
        disp, cost, off, disp_sample, prev_info = self.coarse(left_feat_16, right_feat_16, prev_info)
        low, high = disp - disp_range, disp + disp_range
        disps.append(disp)
        disp_samples.append(disp_sample)
        search_ranges.append({'low': low, 'high': high})
        costs.append(cost)
        offs.append(off)

        # fine prediction
        disp, cost, off, disp_sample, prev_info = self.fine(left_feat_8, right_feat_8, low, high, prev_info)
        low, high = disp - disp_range, disp + disp_range
        disps.append(disp)
        disp_samples.append(disp_sample)
        search_ranges.append({'low': low, 'high': high})
        costs.append(cost)
        offs.append(off)

        # precise
        full_disp, disp, cost, off, disp_sample, prev_info = self.precise(left_feat, right_feat, low, high,
                                                               left_image, right_image, prev_info)
        disps.append(disp)
        disps.append(full_disp)
        disp_samples.append(disp_sample)
        costs.append(cost)
        offs.append(off)

        return disps[::-1], costs[::-1], disp_samples[::-1], offs[::-1], search_ranges[::-1], prev_info

