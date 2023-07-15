import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from .builder import PREDICTION_REGISTRY

@PREDICTION_REGISTRY.register()
class SOFTARGMIN(nn.Module):
    """
    A faster implementation of soft argmin.
    Args:
        temperature,                    (float): a temperature will times with cost_volume, i.e., the temperature coefficient
                                        details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        normalize,                      (bool): whether apply softmax on cost_volume, default True
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
    def __init__(self, temperature:float=1.0, normalize:bool=True):
        super(SOFTARGMIN, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    @classmethod
    def from_config(cls, cfg):
        return {
            "temperature":              cfg.MODEL.PREDICTION.get("TEMPERATURE", 1.0),
            "normalize":                cfg.MODEL.PREDICTION.get("NORMALIZE", True),
        }

    def forward(self, cost_volume, disp_sample):

        # note, cost volume direct represent similarity
        # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        if cost_volume.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(cost_volume.dim()))

        # scale cost volume with temperature
        cost_volume = cost_volume * self.temperature

        if self.normalize:
            prob_volume = F.softmax(cost_volume, dim=1)
        else:
            prob_volume = cost_volume

        assert prob_volume.shape == disp_sample.shape, 'The shape of disparity samples and cost volume should be' \
                                                        ' consistent!'

        # compute disparity: (BatchSize, 1, Height, Width)
        disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)

        return disp_map

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Temperature: {}\n'.format(self.temperature)
        repr_str += ' ' * 4 + 'Normalize: {}\n'.format(self.normalize)

        return repr_str

    @property
    def name(self):
        return 'SoftArgmin'