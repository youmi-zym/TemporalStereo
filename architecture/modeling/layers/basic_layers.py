from typing import Optional, List, Dict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils import env
from detectron2.layers import NaiveSyncBatchNorm, FrozenBatchNorm2d


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return nn.Identity
    if isinstance(norm, str):
        if len(norm) == 0:
            return nn.Identity
        norm = {
            "BN1d": nn.BatchNorm1d,
            "BN": nn.BatchNorm2d,
            "BN3d": nn.BatchNorm3d,
            "IN1d": nn.InstanceNorm1d,
            "IN": nn.InstanceNorm2d,
            "IN3d": nn.InstanceNorm3d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels)


def get_activation(activation, coeff=None):
    """
    Args:
        activation (str or callable): either one of ReLU, ELU and so on;
            or a callable that takes a coefficient and returns
            the activation layer as a nn.Module.
    Returns:
        nn.Module or None: the activation layer
    """
    if activation is None:
        return nn.Identity
    if isinstance(activation, str):
        if len(activation) == 0:
            return nn.Identity

        if activation in ['ELU', 'LeakyReLU'] and coeff is None:
            if activation == 'LeakyReLU':
                coeff = 0.1
            if activation == 'ELU':
                coeff = 1.0
            # raise ValueError("use {}, but coefficient isn't given".format(activation))

        activation = {
            "ReLU": nn.ReLU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(negative_slope=coeff, inplace=True),
            "ELU": nn.ELU(alpha=coeff, inplace=True),
            "SELU": nn.SELU(inplace=True),
            "SiLU": nn.SiLU(inplace=True),
            "Hardswish": nn.Hardswish(inplace=True),
            "Mish": nn.Mish(inplace=True),
        }[activation]
    return activation


def get_norm_and_activation(**kwargs):
    """
    Extra keyword arguments supported in addition to those in `torch.nn.ConvNd`, `torch.nn.ConvTransposeNd`:
    Args:
        norm (nn.Module, optional): a normalization layer
        activation (callable(Tensor) -> Tensor): a callable activation function
    It assumes that norm layer is used before activation.
    """
    norm = kwargs.pop("norm", None)
    if isinstance(norm, (tuple, list)) and len(norm) == 2:
        norm, out_channels = norm
        norm = get_norm(norm, out_channels)

    activation = kwargs.pop("activation", None)
    if isinstance(activation, (tuple, list)):
        if len(activation) == 1:
            activation = activation[0]
            coeff = None
        elif len(activation) == 2:
            activation, coeff = activation
        else:
            raise ValueError
        activation = get_activation(activation, coeff)

    if isinstance(activation, str):
        activation = get_activation(activation, None)

    return norm, activation, kwargs


class Conv1d(torch.nn.Conv1d):
    """
    A wrapper around :class:`torch.nn.Conv1d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv1d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm, activation, kwargs = get_norm_and_activation(**kwargs)

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv1d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm, activation, kwargs = get_norm_and_activation(**kwargs)

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3d(torch.nn.Conv3d):
    """
    A wrapper around :class:`torch.nn.Conv3d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv3d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm, activation, kwargs = get_norm_and_activation(**kwargs)

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTranspose1d(torch.nn.ConvTranspose1d):
    """
    A wrapper around :class:`torch.nn.ConvTranspose1d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.ConvTranspose1d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm, activation, kwargs = get_norm_and_activation(**kwargs)

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `ConvTranspose1d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        x = F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    """
    A wrapper around :class:`torch.nn.ConvTranspose2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.ConvTranspose2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm, activation, kwargs = get_norm_and_activation(**kwargs)

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `ConvTranspose3d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        x = F.conv_transpose2d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTranspose3d(torch.nn.ConvTranspose3d):
    """
    A wrapper around :class:`torch.nn.ConvTranspose3d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.ConvTranspose3d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm, activation, kwargs = get_norm_and_activation(**kwargs)

        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `ConvTranspose3d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        x = F.conv_transpose3d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


if __name__ == '__main__':
    """
    Test ConvNd, ConvTransposeNd
    """
    # test 2d
    a = torch.randn(1, 1, 3, 4)
    print(a)
    conv2d = Conv2d(1, 1, 3, 1, 1, norm=('BN', 1), activation=nn.ReLU(inplace=True))
    print(conv2d)
    b = conv2d(a)
    print(b)
    deconv2d = ConvTranspose2d(1, 1, 4, 2, 1, norm=('BN', 1), activation=('ReLU', ))
    print(deconv2d)
    # output_size can be [6,8] to [7, 9]
    c = deconv2d(b, output_size=(6, 8))
    print(c)
    d = deconv2d(b, output_size=None)
    print(d)
    e = deconv2d(b, output_size=(6, 9))
    print(e)

    # test 3d
    conv3d = Conv3d(1, 1, 3, 1, 1, norm=('BN3d', 1), activation=None)
    print(conv3d)
    deconv3d = ConvTranspose3d(1, 1, 4, 2, 1, norm=('BN3d', 1), activation=("ELU", 1.0))
    print(deconv3d)

    # test 1d
    conv1d = Conv1d(1, 1, 3, 1, 1, norm=nn.BatchNorm1d(1), activation=('ReLU'))
    print(conv1d)

    print("Done!")