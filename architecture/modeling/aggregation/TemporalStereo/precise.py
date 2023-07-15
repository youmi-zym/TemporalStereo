import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from architecture.modeling.aggregation.utils import block_cost

from .module import PredictionHeads, ResidualBlock3D, DepthwiseConv3D, UNet

class PreciseAggregation(nn.Module):
    def __init__(self,
                 in_planes: int,
                 C: int,
                 num_sample: int,
                 delta: float = 1,
                 block_cost_scale: int = 3,
                 topk: int = 2,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List] = 'SiLU'):
        super(PreciseAggregation, self).__init__()
        self.in_planes = in_planes
        self.C = C
        self.num_sample = num_sample
        self.delta = delta
        self.block_cost_scale = block_cost_scale
        self.topk = topk
        self.norm = norm
        self.activation = activation

        cost_planes = 4 * in_planes + block_cost_scale * 2 * in_planes // 8
        self.init3d = nn.Sequential(
            DepthwiseConv3D(cost_planes, C, 3, 1, 1, bias=True, norm=norm, activation=activation),
            ResidualBlock3D(in_planes=C, kernel_size=3, stride=2, padding=1, norm=norm, activation=activation),
            DepthwiseConv3D(C, C, 3, 1, padding=2, dilation=2, bias=False, norm=norm, activation=activation),
        )

        self.pred_heads = PredictionHeads(in_planes=C, delta=delta, norm=norm, activation=activation)

        self.refinement = UNet(in_planes=3, out_planes=in_planes)

        self.weight_init()

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

    def predict_disp(self, cost, disp_sample, off, k=2):
        topk_cost, indices = torch.topk(cost, k=k, dim=1)
        prob = torch.softmax(topk_cost, dim=1)
        topk_disp = torch.gather(disp_sample+off, dim=1, index=indices)
        disp_map = torch.sum(prob*topk_disp, dim=1, keepdim=True)

        return disp_map, topk_disp, topk_cost

    def generate_disparity_sample(self, low_disparity, high_disparity, num_sample):
        batch_size, _, height, width = low_disparity.shape
        device = low_disparity.device
        # track with constant speed motion
        disp_sample = torch.Tensor([0, 3, 4, 5, 8])
        num_sample = len(disp_sample)
        disp_sample = (disp_sample / disp_sample.max()).view(1, num_sample, 1, 1)
        disp_sample = disp_sample.expand(batch_size, num_sample, height, width).to(device)
        disp_sample = torch.abs(high_disparity - low_disparity) * disp_sample + torch.min(low_disparity, high_disparity)

        return disp_sample

    def forward(self, left, right, low_disparity, high_disparity, left_image, right_image, prev_info:dict):
        B, _, H, W = left.shape
        spx_left_feats, spx_right_feats = self.refinement.encoder(left_image, right_image)
        spx2l, spx4l = spx_left_feats
        spx2r, spx4r = spx_right_feats
        left, right = torch.cat([left, spx4l], dim=1), torch.cat([right, spx4r], dim=1)

        disp_sample = self.generate_disparity_sample(low_disparity, high_disparity, self.num_sample)
        raw_cost = block_cost(left, right, disp_sample, block_cost_scale=self.block_cost_scale)

        init_cost = self.init3d(raw_cost)
        final_cost, off = self.pred_heads(init_cost)

        # learn disparity
        disp, memory_sample, memory_volume = self.predict_disp(final_cost, disp_sample, off, k=self.topk)
        full_disp = self.refinement.decoder(disp, left, spx2l)

        prev_info['prev_disp'] = full_disp.detach()
        # save memory
        prev_info['cost_memory'] = {
            'disp_sample': F.interpolate(memory_sample/2, scale_factor=1/2, mode='bilinear', align_corners=True),
            'cost_volume': F.interpolate(memory_volume, scale_factor=1/2, mode='bilinear', align_corners=True),
        }

        return full_disp, disp, final_cost, off, disp_sample, prev_info
