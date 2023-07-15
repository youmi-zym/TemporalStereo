import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from .builder import PREDICTION_REGISTRY

@PREDICTION_REGISTRY.register()
class ARGMIN(nn.Module):
    """
    A faster implementation of argmin.
    Args:
        dim,                            (int): perform argmin at dimension $dim

    Inputs:
        cost_volume,                    (Tensor): the matching cost after regularization,
                                        [BatchSize, disp_sample_number, Height, Width] layout
        disp_sample,                    (Tensor): the estimated disparity samples,
                                        [BatchSize, disp_sample_number, Height, Width] layout.
    Returns:
        disp_map,                       (Tensor): a disparity map regressed from cost volume,
                                        [BatchSize, 1, Height, Width] layout
    """
    @configurable
    def __init__(self, dim:int = 1):
        super(ARGMIN, self).__init__()
        self.dim = dim

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim":                      cfg.MODEL.PREDICTION.get("DIM", 1),
        }

    def forward(self, cost_volume, disp_sample):

        # note, cost volume direct represent similarity
        # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        assert cost_volume.shape == disp_sample.shape, "{}, {}".format(cost_volume.shape, disp_sample.shape)

        _, indices = torch.max(cost_volume, dim=self.dim, keepdim=True)
        # compute disparity: (BatchSize, 1, Height, Width)
        disp_map = torch.gather(disp_sample, dim=self.dim, index=indices)

        return disp_map

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Dim: {}\n'.format(self.dim)

        return repr_str

    @property
    def name(self):
        return 'Argmin'