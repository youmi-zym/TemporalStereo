import torch
import matplotlib.pyplot as plt
from typing import Union

def colormap(_cmap, *args, normalize:bool=True, format:str='HWC', **kwargs):
    """
    color a given array
    Args:
        _cmap:                          (str, callable function), the colormap or direct give function handle
        normalize:                      (bool), whether perform max-min-normalization
        format:                         (str), the format of output colormap,
                                        if 'CHW' -> [Channel, Height, Width], elif 'HWC' -> [Height, Width, Channel]
        args:                           (Tensor, numpy.array), the inputs is required in cmap function
                                        [..., H, W]
        kwargs:                         (dict), args required in cmap function

    Outputs:

    """
    inputs = []
    for input in args:
        if isinstance(input, torch.Tensor):
            input = input.detach().cpu().numpy()

        if normalize:
            ma = float(input.max())
            mi = float(input.min())
            d = ma - mi if ma != mi else 1e5
            input = (input - mi) / d

        if input.ndim == 4:
            if input.shape[1] == 1:
                # one channel map
                input = input[0, 0]
            elif input.shape[1] == 2:
                # here, we fix it as flow
                input = input[0]
                input = input.transpose(1, 2, 0)
            else:
                input = input[0]

        elif input.ndim == 3:
            if input.shape[0] == 1:
                input = input[0]
            elif input.shape[0] == 2:
                input = input.transpose(1, 2, 0)
            else:
                pass

        elif input.ndim == 2:
            pass
        else:
            assert input.ndim > 1 and input.ndim < 5, input.ndim

        inputs.append(input)

    if isinstance(_cmap, str):
        if _cmap == 'plasma':
            _COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
        elif _cmap == 'gray':
            _COLORMAP = plt.get_cmap('gray')
        elif _cmap == 'jet':
            _COLORMAP = plt.get_cmap('jet')
        else:
            raise ValueError("invalid type: {} received, only support ['jet', 'plasma', 'gray']".format(cmap))

        map = _COLORMAP(*inputs, **kwargs)

    elif callable(_cmap):
        map = _cmap(*inputs, **kwargs)

    else:
        raise ValueError(_cmap)

    # [H, W, C], value range [0, 1]
    map = map[..., :3]

    if format == 'HWC':
        pass
    elif format == 'CHW':
        map = map.transpose(2, 0, 1)
    else:
        raise ValueError(format)

    return map