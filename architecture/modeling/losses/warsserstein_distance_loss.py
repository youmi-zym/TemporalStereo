import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from typing import Optional, Dict, Tuple, Union, List
import warnings


class WarssersteinDistanceLoss(object):
    """
    Args:
        max_disp:                       (int), the max of Disparity. default is 192
        start_disp:                     (int), the start searching disparity index, usually be 0
        weights:                        (list, tuple, float, optional): weight for each scale of est disparity map.
        sparse:                         (bool), whether the ground-truth disparity is sparse,
                                        for example, KITTI is sparse, but SceneFlow is not, default is False.
    Inputs:
        estCosts:                       (Tensor or List[Tensor]): the estimated cost volumes,
                                        [BatchSize, NumSamples, Height, Width] layout.
        estOffsets:                     (Tensor or List[Tensor]): the estimated disparity offsets for each cost volume,
                                        [BatchSize, NumSamples, Height, Width] layout.
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
            "global_weight":            cfg.get("GLOBAL_WEIGHT", 1.0),
            "weights":                  cfg.get("WEIGHTS", None),
            "sparse":                   cfg.get("SPARSE", False),
        }

    def loss_per_level(self, estCost, estOffset, dispSample, gtDisp):
        N, D, H, W = estCost.shape
        estProb = torch.softmax(estCost, dim=1)

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
            warnings.warn('Warsserstein distance loss: there is no point\'s disparity is in ({},{})!'.format(self.start_disp,
                                                                                        self.max_disp / scale))
            loss = (estProb * torch.abs(estOffset + dispSample - scaled_gtDisp) * mask.float()).sum(dim=1).mean()
            return loss

        # warsserstein distance loss
        # sum{ (0.2 + 0.8*P(d)) * | d* - (d + delta)| }
        war_loss = ((estProb*1.0 + 0.25) * torch.abs(estOffset + dispSample - scaled_gtDisp) * mask.float()).sum(dim=1).mean()

        return war_loss

    def __call__(self, estCosts, estOffsets, dispSamples, gtDisp):
        if not isinstance(estCosts, (list, tuple)):
            estCosts = [estCosts, ]

        if not isinstance(estOffsets, (list, tuple)):
            estOffsets = [estOffsets, ]

        if not isinstance(dispSamples, (list, tuple)):
            dispSamples = [dispSamples, ] * len(estCosts)

        assert len(estCosts) == len(estOffsets), "{}, {}".format(len(estCosts), len(estOffsets))

        if self.weights is None:
            self.weights = [1.0] * len(estCosts)

        # compute loss for per level
        loss_all_level = []
        for est_cost_per_lvl, est_off_per_lvl, est_sample_per_lvl in zip(estCosts, estOffsets, dispSamples):
            assert est_sample_per_lvl.shape == est_cost_per_lvl.shape, "sample shape: {}, cost shape: {}".format(est_sample_per_lvl.shape,
                                                                                                                 est_cost_per_lvl.shape)
            assert est_off_per_lvl.shape == est_cost_per_lvl.shape, "sample shape: {}, cost shape: {}".format(est_off_per_lvl.shape,
                                                                                                              est_cost_per_lvl.shape)

            loss_all_level.append(
                self.loss_per_level(est_cost_per_lvl, est_off_per_lvl, est_sample_per_lvl, gtDisp)
            )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "wars_loss_lvl{}".format(i)
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
        return 'WarssersteinDistanceLoss'