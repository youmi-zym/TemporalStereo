import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from architecture.modeling.layers import Conv3d

class SPP3D(nn.Module):
    """
    3D SPP
    Args:
        in_planes:                      (int), the channels of feature map
        norm:                           (str), the type of normalization layer
        activation:                     (str, list, tuple), the type of activation layer and its coefficient is needed
    """
    def __init__(self,
                 in_planes: int = 64,
                 strides: Union[List, Tuple] = (2, 4, 8, 16),
                 norm: str = 'BN3d',
                 activation: Union[str, List, Tuple] = 'ReLU'):
        super(SPP3D, self).__init__()

        self.in_planes = in_planes
        self.strides = strides
        self.norm = norm
        self.activation = activation

        self.pools = nn.ModuleList()
        for stride in self.strides:
            self.pools.append(
                Conv3d(in_planes, 16, 1, 1, 0, 1, bias=False, norm=(norm, 16), activation=activation)
            )
        self.fuse = nn.Sequential(
            Conv3d(16*len(strides)+in_planes, in_planes, 3, 1, 1, 1, bias=False, norm=(norm, in_planes), activation=activation),
            nn.Conv3d(in_planes, in_planes, 1, 1, 0, 1, bias=False),
        )

    def forward(self, x):
        features = [x]
        B, C, D, H, W = x.shape
        for stride, pool in zip(self.strides, self.pools):
            stride = (min(D, stride), min(H, stride), min(W, stride))
            out = F.avg_pool3d(x, kernel_size=stride, stride=stride)
            out = pool(out)
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=True)
            features.append(out)
        features = torch.cat(features, dim=1)
        out = self.fuse(features)
        return out

