import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import math

from architecture.modeling.layers import Conv2d, Conv3d, ConvTranspose3d, ConvTranspose2d, get_norm

class ResidualBlock2D(nn.Module):
    """
    An implementation of residual block in 2D.
    Reference forward once takes 1.1056ms, i.e. 904.51fps at 1/16 with in_planes=32
    Reference forward once takes 1.1129ms, i.e. 898.58fps at 1/ 8 with in_planes=32
    Reference forward once takes 1.2657ms, i.e. 790.08fps at 1/ 4 with in_planes=32
    Args:
        in_planes:                      (int), the channels of raw cost volume
        integrate_last_stage:           (int), the channels of Tensor from lower/last stage, if 0, means None
        norm:                           (str), the type of normalization layer
        activation:                     (str, list, tuple), the type of activation layer and its coefficient is needed
    Inputs:
        x:                              (Tensor): cost volume
                                        [BatchSize, in_planes, Height, Width] layout
        last_stage:                     (Optional, Tensor): the Tensor from lower/last stage
                                        [BatchSize, integrate_last_stage, Height//2, Width//2] layout
    Outputs:
        out:                            (Tensor), cost volume
                                        [BatchSize, in_planes, MaxDisparity, Height, Width] layout
    """

    def __init__(self,
                 in_planes: int,
                 norm: str = 'BN',
                 activation: Union[str, Tuple, List] = 'SiLU'):
        super(ResidualBlock2D, self).__init__()
        self.in_planes = in_planes
        self.norm = norm
        self.activation = activation

        self.conv1 = Conv2d(
            in_planes, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False,
            norm=(norm, in_planes * 2), activation=activation
        )

        self.conv2 = Conv2d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False,
            norm=(norm, in_planes * 2), activation=activation
        )

        self.conv3 = Conv2d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False,
            norm=(norm, in_planes * 2), activation=activation
        )

        self.conv4 = Conv2d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False,
            norm=(norm, in_planes * 2), activation=activation
        )

        self.conv5 = ConvTranspose2d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False,
            norm=(norm, in_planes * 2), activation=None
        )
        self.conv6 = ConvTranspose2d(
            in_planes * 2, in_planes,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False,
            norm=(norm, in_planes), activation=None
        )

        self.shortcut5 = Conv2d(
            in_planes * 2, in_planes * 2,
            kernel_size=1, stride=1, padding=0, bias=False,
            norm=(norm, in_planes * 2), activation=None
        )

        self.shortcut6 = Conv2d(
            in_planes, in_planes,
            kernel_size=1, stride=1, padding=0, bias=False,
            norm=(norm, in_planes), activation=None
        )

    def forward(self, x):
        # in: [B, C, H, W], out: [B, 2C, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, H/2, W/2], out: [B, 2C, H/2, W/2]
        pre = self.conv2(out)

        # in: [B, 2C, H/2, W/2], out: [B, 2C, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, H/4, W/4], out: [B, 2C, H/4, W/4]
        out = self.conv4(out)

        # in: [B, 2C, H/4, W/4], out: [B, 2C, H/2, W/2]
        H, W = pre.shape[-2:]
        out = self.conv5(out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        out = F.silu(out + self.shortcut5(pre), inplace=True)

        # in: [B, 2C, H/2, W/2], out: [B, C, H, W]
        out = self.conv6(out)
        H, W = x.shape[-2:]
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        out = F.silu(out + self.shortcut6(x), inplace=True)

        return out

class DepthwiseConv3D(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int = 1,
                 bias: bool = False,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List, None] = 'SiLU'):
        super(DepthwiseConv3D, self).__init__()

        self.conv = nn.Sequential(
            Conv3d(
                in_planes, out_planes,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                dilation=(1, dilation, dilation),
                bias=bias,
                norm=(norm, out_planes), activation=activation
            ),

            Conv3d(
                out_planes, out_planes,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                dilation=(dilation, 1, 1),
                bias=bias,
                norm=(norm, out_planes), activation=activation
            )
        )

    def forward(self, x):
        return self.conv(x)

class DepthwiseConvTranspose3D(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int,
                 bias: bool = False,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List, None] = 'SiLU'):
        super(DepthwiseConvTranspose3D, self).__init__()

        self.conv = nn.Sequential(
            ConvTranspose3d(
                in_planes, out_planes,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                output_padding=(0, output_padding, output_padding),
                bias=bias,
                norm=(norm, out_planes), activation=activation
            ),
            ConvTranspose3d(
                out_planes, out_planes,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                output_padding=(output_padding, 0, 0),
                bias=bias,
                norm=(norm, out_planes), activation=activation
            )
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock3D(nn.Module):
    """
    An implementation of 3D residual block
    Args:
        in_planes:                      (int), the channels of raw cost volume
        kernel_size:                    (int, tuple), the kernel size of convolution layer
        stride:                         (int, tuple), the stride for downsampling
        padding:                        (int, tuple), the padding size of convolution layer
        norm:                           (str), the type of normalization layer
        activation:                     (str, list, tuple), the type of activation layer and its coefficient is needed
    Inputs:
        x:                              (Tensor): cost volume
                                        [BatchSize, in_planes, MaxDisparity, Height, Width] layout
    Outputs:
        out:                            (Tensor), cost volume
                                        [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        pre:                            (optional, Tensor), cost volume
                                        [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        post:                           (optional, Tensor), cost volume
                                        [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
    """

    def __init__(self,
                 in_planes: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List] = 'SiLU'):
        super(ResidualBlock3D, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm = norm
        self.activation = activation

        self.conv1 = DepthwiseConv3D(
            in_planes, in_planes * 2,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
            norm=norm, activation=activation
        )

        self.conv2 = DepthwiseConv3D(
            in_planes * 2, in_planes * 2,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False,
            norm=norm, activation=activation
        )

        self.conv3 = DepthwiseConv3D(
            in_planes * 2, in_planes * 2,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
            norm=norm, activation=activation
        )

        self.conv4 = DepthwiseConv3D(
            in_planes * 2, in_planes * 2,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False,
            norm=norm, activation=None,
        )

        self.conv5 = DepthwiseConvTranspose3D(
            in_planes * 2, in_planes * 2,
            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding, bias=False,
            norm=norm, activation=None
        )

        self.conv6 = DepthwiseConvTranspose3D(
            in_planes * 2, in_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding, bias=False,
            norm=norm, activation=None
        )

        self.shortcut5 = DepthwiseConv3D(
            in_planes * 2, in_planes * 2,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False,
            norm=norm, activation=None
        )

        self.shortcut6 = DepthwiseConv3D(
            in_planes, in_planes,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False,
            norm=norm, activation=None
        )

    def forward(self, x):
        # in: [B, C, D, H, W], out: [B, 2C, D/2, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, D/2, H/2, W/2], out: [B, 2C, D/2, H/2, W/2]
        pre = self.conv2(out)

        # in: [B, 2C, D/2, H/2, W/2], out: [B, 2C, D/4, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, D/4, H/4, W/4], out: [B, 2C, D/4, H/4, W/4]
        out = self.conv4(out)
        out = F.silu(out, inplace=True)

        # in: [B, 2C, D/4, H/4, W/4], out: [B, 2C, D/2, H/2, W/2]
        D, H, W = pre.shape[-3:]
        out = self.conv5(out)
        out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=True)
        out = F.silu(out + self.shortcut5(pre), inplace=True)


        # in: [B, 2C, D, H/2, W/2], out: [B, C, D, H, W]
        out = self.conv6(out)
        D, H, W = x.shape[-3:]
        out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=True)
        out = F.silu(out + self.shortcut6(x), inplace=True)

        return out


class ConvexUpsample(nn.Module):
    def  __init__(self,
                  in_planes: int,
                  upscale_factor: int = 2,
                  window_size: int = 3):
        super(ConvexUpsample, self).__init__()
        self.in_planes = in_planes
        self.upscale_factor = upscale_factor
        self.window_size = window_size

        self.mask = nn.Sequential(
            nn.Conv2d(in_planes, 64, (3,3), (1,1), (1,1), bias=True),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, (window_size ** 2) * (upscale_factor ** 2),
                      kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=0, bias=True),
        )

    def forward(self, input, disp, disp_scale=None):
        """
        upsample disparity map to up resolution
        Args:
            input:                      (Tensor): tensor used to generate mask weight
                                        [BatchSize, in_planes, Height, Width]
            disp:                       (Tensor): low resolution disparity map
                                        [BatchSize, 1, Height, Width]
            upscale_factor:             (int, float): the upsample scale factor
        Return:
            up_disp:                    (Tensor): upsampled disparity map
                                        [BatchSize, 1, Height*upscale_factor, Width*upscale_factor]
        """
        B, C, H, W = disp.shape

        # assert C == 1
        assert self.window_size % 2 == 1, "{}".format(self.window_size)

        mask = self.mask(input)
        mask = mask.view(B, 1, self.window_size**2, self.upscale_factor, self.upscale_factor, H, W)
        mask = torch.softmax(mask, dim=2)

        if disp_scale is None:
            disp_scale = self.upscale_factor

        up_disp = F.unfold(disp * disp_scale,
                           kernel_size=(self.window_size, self.window_size),
                           padding=(self.window_size//2, self.window_size//2))
        up_disp = up_disp.view(B, C, self.window_size**2, 1, 1, H, W)

        # [B, C, 8, 8, H, W]
        up_disp = torch.sum(mask * up_disp, dim=2, keepdim=False)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3).contiguous()
        up_disp = up_disp.reshape(B, C, H * self.upscale_factor, W * self.upscale_factor)

        return up_disp


class PredictionHeads(nn.Module):
    def __init__(self,
                 in_planes: int,
                 delta: float = 1,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List] = 'SiLU',):
        super(PredictionHeads, self).__init__()
        self.in_planes = in_planes
        self.delta = delta
        self.norm = norm
        self.activation = activation

        self.cost_head = nn.Sequential(
            Conv3d(in_planes, in_planes, (3, 1, 1), 1, (1, 0, 0), bias=False, norm=(norm, in_planes),
                   activation=activation),
            Conv3d(in_planes, 1, (1, 3, 3), 1, (0, 1, 1), bias=False, norm=None, activation=None),
        )

        self.off_head = nn.Sequential(
            Conv3d(in_planes, in_planes, (3, 1, 1), 1, (1, 0, 0), bias=False, norm=(norm, in_planes),
                   activation=activation),
            Conv3d(in_planes, 1, (1, 3, 3), 1, (0, 1, 1), bias=False, norm=None, activation=None),
        )

    def regress_offset(self, off):
        # [-1, 1], soften the value space of tanh, like a temperature coefficient in softmax
        off = torch.tanh(off / 100).clamp(-1, 1)
        # [-delta, delta]
        off = off * self.delta

        return off

    def forward(self, init_cost):
        # learn offset
        off = self.off_head(init_cost)
        off = self.regress_offset(off)
        off = off.squeeze(dim=1)

        # learn cost
        cost = self.cost_head(init_cost)
        cost = cost.squeeze(dim=1)

        return cost, off


class PyramidFusion(nn.Module):
    def __init__(self,
                 in_planes: int,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List] = 'SiLU',):
        super(PyramidFusion, self).__init__()

        self.conv_5x5 = Conv3d(in_planes, in_planes, (5, 1, 1), 1, (2, 0, 0), bias=False, norm=('BN3d', in_planes), activation=activation)
        self.conv_fuse = DepthwiseConv3D(4*in_planes, in_planes, kernel_size=3, stride=1, padding=1,
                                         bias=False, norm=norm, activation=None)

    def forward(self, cost):
        cost = torch.cat([
            cost,
            self.conv_5x5(cost),
            F.avg_pool3d(cost, kernel_size=5, stride=1, padding=2),
            F.max_pool3d(cost, kernel_size=5, stride=1, padding=2),
        ], dim=1)
        cost = self.conv_fuse(cost)

        return cost


class UNet(nn.Module):
    def __init__(self,
                 in_planes: int = 3,
                 out_planes: int = 48,
                 norm: str = 'BN',
                 activation: str = 'SiLU'):
        super(UNet, self).__init__()
        self.in_planes = in_planes
        C = 32
        activation = 'ReLU'

        self.conv2 = nn.Sequential(
            Conv2d(in_planes, C, kernel_size=3, stride=2, padding=1, bias=False,
                   norm=(norm, C), activation=activation),
            Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False,
                   norm=(norm, C), activation=activation),
        )
        self.conv4 = nn.Sequential(
            Conv2d(C, out_planes, kernel_size=3, stride=2, padding=1, bias=False,
                   norm=(norm, out_planes), activation=activation),
            Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                   norm=(norm, out_planes), activation=activation),
        )

        self.fuse = nn.Sequential(
            Conv2d(out_planes*2, C, kernel_size=3, stride=1, padding=1, bias=False,
                   norm=(norm, C), activation=activation),
            Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False,
                   norm=(norm, C), activation=activation),
        )
        self.deconv4 = ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1, norm=(norm, C), activation=activation)
        self.concat = Conv2d(C*2, C, kernel_size=3, stride=1, padding=1, bias=False,
                             norm=(norm, C), activation=activation)
        self.deconv2 = nn.ConvTranspose2d(C, 9, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def encoder(self, imL, imR):
        spx2l = self.conv2(imL)
        spx4l = self.conv4(spx2l)

        spx2r = self.conv2(imR)
        spx4r = self.conv4(spx2r)

        return [spx2l, spx4l], [spx2r, spx4r]

    def upsample(self, mask, disp):
        # [B, 9, H, W]
        mask = F.softmax(mask, dim=1)

        b, _, h, w = mask.shape
        _, _, dh, dw = disp.shape

        disp = F.unfold(disp, kernel_size=(3, 3), padding=(1, 1))
        # [B, 9, H//x, W//x]
        disp = disp.reshape(b, 9, dh, dw)
        # [B, 9, H, W]
        full_disp = F.interpolate(disp*w/dw, size=(h, w), mode='bilinear', align_corners=True)
        # [B, 1, H, W]
        full_disp = torch.sum(full_disp * mask, dim=1, keepdim=True)

        return full_disp

    def decoder(self, disp, feat, feat2x):
        feat = self.fuse(feat)
        feat = self.deconv4(feat)
        feat = self.concat(torch.cat([feat, feat2x], dim=1))
        mask = self.deconv2(feat)
        full_disp = self.upsample(mask, disp)

        return full_disp


from architecture.modeling.layers import inverse_warp
class StereoDRNetRefinement(nn.Module):
    def __init__(self):
        super(StereoDRNetRefinement, self).__init__()
        C = 16
        self.feat_conv = Conv2d(4*3, C, 3, 1, 1, bias=False, norm=('BN', C), activation='ReLU')
        self.disp_conv = Conv2d(1, C, 3, 1, 1, bias=False, norm=('BN', C), activation='ReLU')
        self.dilated_block = nn.Sequential(*[
            BasicBlock(C*2, C*2, stride=1, dilation=dilation) for dilation in [1, 2, 4, 8, 1, 1]
        ])

        self.final_conv = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, disp, left_image, right_image):
        warp_left = inverse_warp(right_image, -disp, mode='disparity')
        error = torch.abs(warp_left - left_image)
        feat = self.feat_conv(torch.cat([left_image, right_image, warp_left, error], dim=1))
        feat = self.dilated_block(torch.cat([feat, self.disp_conv(disp)], dim=1))
        res = self.final_conv(feat)
        disp = F.relu(disp + res, inplace=True)
        return disp
        

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride, downsample=None, padding=1, dilation=1, norm='BN', activation='ReLU'):
        super(BasicBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        self.conv1 = Conv2d(in_planes,  out_planes, 3, stride, padding, dilation, bias=False, norm=(norm, out_planes), activation=activation)
        self.conv2 = Conv2d(out_planes, out_planes, 3, 1,      padding, dilation, bias=False, norm=(norm, out_planes), activation=None)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out
