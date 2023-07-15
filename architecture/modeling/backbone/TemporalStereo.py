import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Tuple
import math
import timm
from timm.models.efficientnet_blocks import InvertedResidual, drop_path

from detectron2.config import configurable

from .backbone import Backbone
from .builder import BACKBONE_REGISTRY

from architecture.modeling.layers import Conv2d, ConvTranspose2d


@BACKBONE_REGISTRY.register()
class TEMPORALSTEREO(Backbone):
    """
    Backbone proposed in COEX.
    Args:
        in_planes:                      (int), the channels of input image
        norm:                           (str), the type of normalization layer
        activation:                     (str, list, tuple), the type of activation layer and its coefficient is needed
    Inputs:
        l_img,                          (Tensor): left image
                                        [BatchSize, 3, Height, Width] layout
        r_img,                          (Tensor): right image
                                        [BatchSize, 3, Height, Width] layout
    Outputs:
        l_fms,                          (Tensor), left image feature maps
                                        [
                                            [BatchSize, 48, Height//4, Width//4],
                                            [BatchSize, 64, Height//8, Width//8],
                                            [BatchSize, 192, Height//16, Width//16],
                                            [BatchSize, 160, Height//32, Width//32],
                                        ]
        r_fms,                          (Tensor), right image feature maps
                                        [
                                            [BatchSize, 48, Height//4, Width//4],
                                            [BatchSize, 64, Height//8, Width//8],
                                            [BatchSize, 192, Height//16, Width//16],
                                            [BatchSize, 160, Height//32, Width//32],
                                        ]
    """
    @configurable
    def __init__(self,
                 in_planes:int = 3,
                 memory_percent: float = 1/4,
                 norm: str = 'BN',
                 activation: Union[str, list, tuple] = 'SiLU'):
        super(TEMPORALSTEREO, self).__init__()

        self.in_planes              = in_planes
        self.memory_percent         = memory_percent
        self.norm                   = norm
        self.activation             = activation

        net = timm.create_model('efficientnetv2_rw_s', pretrained=True)

        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1
        self.act1 = net.act1

        layers = [1, 2, 3, 5, 7,]

        self.block0 = torch.nn.Sequential(*net.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*net.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*net.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*net.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*net.blocks[layers[3]:layers[4]])

        channels = [24, 48, 64, 160, 272]
        out_channels = [0, 64, 128, 256, 320]


        self.conv32 = Conv2d(channels[4], out_channels[4], 3, 1, 1, bias=False, norm=None, activation=None)
        self.deconv32_16 = nn.Sequential(
            Conv2d(out_channels[4]+channels[3], out_channels[3], 3, 1, 1, bias=False, norm=(norm, out_channels[3]), activation=activation),
            Conv2d(out_channels[3], out_channels[3], 3, 1, 1, bias=False, norm=None, activation=None)
        )
        self.deconv16_8  = nn.Sequential(
            Conv2d(out_channels[3]+channels[2], out_channels[2], 3, 1, 1, bias=False, norm=(norm, out_channels[2]), activation=activation),
            Conv2d(out_channels[2], out_channels[2], 3, 1, 1, bias=False, norm=None, activation=None)
        )
        self.deconv8_4   = nn.Sequential(
            Conv2d(out_channels[2]+channels[1], out_channels[1], 3, 1, 1, bias=False, norm=(norm, out_channels[1]), activation=activation),
            Conv2d(out_channels[1], out_channels[1], 3, 1, 1, bias=False, norm=None, activation=None)
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_planes":            cfg.MODEL.BACKBONE.get('IN_PLANES', 3),
            "memory_percent":       cfg.MODEL.BACKBONE.get('MEMORY_PERCENT', 1/4),
            "norm":                 cfg.MODEL.BACKBONE.get('NORM', 'BN'),
            "activation":           cfg.MODEL.BACKBONE.get('ACTIVATION', 'SiLU'),
        }

    def _forward(self, x, memories:Union[None, List[torch.Tensor], Tuple[torch.Tensor]]):
        memory_idx = 0
        out_memories = []
        # [B, 32, H//2, W//2]
        x = self.act1(self.bn1(self.conv_stem(x)))
        # [B, 16, H//2, W//2]
        x = self.block0(x)
        # [B, 24, H//4, W//4]
        x, out, memory_idx = _block_forward(self.block1, x, self.memory_percent, memories, memory_idx)
        x4 = x
        out_memories.extend(out)
        # [B, 32, H//8, W//8]
        x, out, memory_idx = _block_forward(self.block2, x, self.memory_percent, memories, memory_idx)
        x8 = x
        out_memories.extend(out)
        # [B, 96, H//16, W//16]
        x, out, memory_idx = _block_forward(self.block3, x, self.memory_percent, memories, memory_idx)
        x16 = x
        out_memories.extend(out)
        # [B, 160, H//32, W//32]
        x, out, memory_idx = _block_forward(self.block4, x, self.memory_percent, memories, memory_idx)
        x32 = x
        out_memories.extend(out)

        # [B, 96, H//32, W//32]
        x32 = self.conv32(x32)
        # [B, 80, H//16, W//16]
        h, w = x16.shape[-2:]
        up_x32 = F.interpolate(x32, size=(h, w), mode='bilinear', align_corners=True)
        x16 = self.deconv32_16(torch.cat([up_x32, x16], dim=1))
        # [B, 64, H//8, W//8]
        h, w = x8.shape[-2:]
        up_x16 = F.interpolate(x16, size=(h, w), mode='bilinear', align_corners=True)
        x8 = self.deconv16_8(torch.cat([up_x16, x8], dim=1))
        # [B, 48, H//4, W//4]
        h, w = x4.shape[-2:]
        up_x8 = F.interpolate(x8, size=(h, w), mode='bilinear', align_corners=True)
        x4 = self.deconv8_4(torch.cat([up_x8, x4], dim=1))

        return [x4, x8, x16], out_memories

    def forward(self, *input):
        if len(input) != 3:
            raise ValueError('expected input length 3 (got {} length input)'.format(len(input)))

        l_img, r_img, prev_info = input
        mem = prev_info.get('memories', [])
        B, _, _, _ = l_img.shape

        lr_img = torch.cat([l_img, r_img], dim=0)

        lr_fms, mem = self._forward(lr_img, mem)

        l_fms = [fms[:B] for fms in lr_fms]
        r_fms = [fms[B:] for fms in lr_fms]

        if self.memory_percent > 0:
            prev_info['memories'] = mem
        else:
            prev_info['memories'] = []

        return l_fms, r_fms, prev_info


def _block_forward(block, input, memory_percent=0.0, memories=None, memory_idx=0):
    out_memories = []
    for sequential in block:
        for _InvertedResidual in sequential:
            # assert isinstance(_InvertedResidual, InvertedResidual), "{} got!".format(type(_InvertedResidual))
            if memory_percent > 0 and isinstance(_InvertedResidual, InvertedResidual) and _InvertedResidual.has_residual:
                if memories is not None and len(memories) > 0:
                    m = memories[memory_idx]
                else:
                    m = None
                input, m = _inverted_residual_forward(_InvertedResidual, input, m, memory_percent)
                out_memories.append(m)
                memory_idx += 1
            else:
                input = _InvertedResidual(input)
    return input, out_memories, memory_idx


def _inverted_residual_forward(_InvertedResidual, input, memory=None, memory_percent:float=-1.0):
    ic = input.shape[1]
    if memory is not None:
        mc = memory.shape[1]
        assert mc == int(ic * memory_percent), \
            "input shape: {}; memory shape: {}!".format(input.shape, memory.shape)
    else:
        mc = int(ic * memory_percent)
        memory = None

    input1, input2 = input[:, :mc], input[:, mc:]
    if memory is None:
        memory = input1

    x = torch.cat([memory, input2], dim=1)
    # Point-wise expansion
    x = _InvertedResidual.conv_pw(x)
    x = _InvertedResidual.bn1(x)
    x = _InvertedResidual.act1(x)

    # Depth-wise convolution
    x = _InvertedResidual.conv_dw(x)
    x = _InvertedResidual.bn2(x)
    x = _InvertedResidual.act2(x)

    # Squeeze-and-excitation
    x = _InvertedResidual.se(x)

    # Point-wise linear projection
    x = _InvertedResidual.conv_pwl(x)
    x = _InvertedResidual.bn3(x)

    if _InvertedResidual.drop_path_rate > 0:
        x = drop_path(x, _InvertedResidual.drop_path_rate, _InvertedResidual.training)

    return input+x, input1

