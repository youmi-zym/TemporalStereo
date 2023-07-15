import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from typing import Optional, Dict, Tuple, Union, List
import warnings


class DispSmoothL1Loss(object):
    """
    Args:
        max_disp:                       (int), the max of Disparity. default is 192
        start_disp:                     (int), the start searching disparity index, usually be 0
        weights:                        (list, tuple, float, optional): weight for each scale of est disparity map.
        sparse:                         (bool), whether the ground-truth disparity is sparse,
                                        for example, KITTI is sparse, but SceneFlow is not, default is False.
    Inputs:
        estDisp:                        (Tensor or List[Tensor]): the estimated disparity maps,
                                        [BatchSize, 1, Height, Width] layout.
        gtDisp:                         (Tensor), the ground truth disparity map,
                                        [BatchSize, 1, Height, Width] layout.
    Outputs:
        loss:                           (dict), the loss of each level
    """
    @configurable
    def __init__(self, max_disp:int, start_disp:int=0, global_weight:float=1.0,
                 weights:Union[Tuple, List, float, None]=None, sparse:bool=False):
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.global_weight = global_weight
        self.weights = weights
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    @classmethod
    def from_config(cls, cfg):
        return {
            "max_disp":                 cfg.get("MAX_DISP", 192),
            "start_disp":               cfg.get("START_DISP", 0),
            "weights":                  cfg.get("WEIGHTS", None),
            "sparse":                   cfg.get("SPARSE", False),
        }

    def loss_per_level(self, estDisp, gtDisp):
        N, C, H, W = estDisp.shape
        scaled_gtDisp = gtDisp
        scale = 1.0
        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale per level and scale gtDisp
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp / scale
            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))

        # mask for valid disparity
        # (start disparity, max disparity / scale)
        # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
        mask = (scaled_gtDisp > self.start_disp) & (scaled_gtDisp < (self.max_disp / scale))
        if mask.sum() < 1.0:
            warnings.warn('SmoothL1 loss: there is no point\'s disparity is in ({},{})!'.format(self.start_disp,
                                                                                        self.max_disp / scale))
            loss = (torch.abs(estDisp - scaled_gtDisp) * mask.float()).mean()
            return loss

        # smooth l1 loss
        loss = F.smooth_l1_loss(estDisp[mask], scaled_gtDisp[mask], reduction='mean')

        return loss

    def __call__(self, estDisp, gtDisp):
        if not isinstance(estDisp, (list, tuple)):
            estDisp = [estDisp]

        if self.weights is None:
            self.weights = [1.0] * len(estDisp)

        # compute loss for per level
        loss_all_level = []
        for est_disp_per_lvl in estDisp:
            loss_all_level.append(
                self.loss_per_level(est_disp_per_lvl, gtDisp)
            )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "l1_loss_lvl{}".format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level * self.global_weight

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Global Loss weight: {}\n'.format(self.global_weight)
        repr_str += ' ' * 4 + 'Loss weights: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'GT Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'SmoothL1Loss'