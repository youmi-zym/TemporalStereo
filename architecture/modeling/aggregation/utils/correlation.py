import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except:
    pass


def correlation(reference_fm, target_fm, patch_size=1,
                kernel_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):
    # for a pixel of left image at (x, y), it will calculates correlation cost volume
    # with pixel of right image at (xr, y), where xr in [x-(patch_size-1)//2, x+(patch_size-1)//2]
    correlation_sampler = SpatialCorrelationSampler(kernel_size= kernel_size,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    dilation_patch=dilation_patch)
    # [B, pH, pW, H, W]
    out = correlation_sampler(reference_fm, target_fm)

    B, pH, pW, H, W = out.shape

    out = out.reshape(B, pH * pW, H, W)

    cost = F.leaky_relu(out, negative_slope=0.1, inplace=True)

    return cost


def correlation1d(reference_fm, target_fm, max_disp=1,
                  kernel_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):

    patch_size = (1, 2 * max_disp - 1)
    # for a pixel of left image at (x, y), it will calculates correlation cost volume
    # with pixel of right image at (xr, y), where xr in [x-max_disp, x+max_disp]
    # but we only need the left half part, i.e., [x-max_disp, 0]
    correlation_sampler = SpatialCorrelationSampler(kernel_size= kernel_size,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    dilation_patch=dilation_patch)
    # [B, 1, 2*max_disp-1, H, W]
    out = correlation_sampler(reference_fm, target_fm)

    B, pH, pW, H, W = out.shape

    out = out.reshape(B, pH * pW, H, W)

    out = out[:, :max_disp, :, :]

    # [B, max_disp, H, W]
    cost = F.leaky_relu(out, negative_slope=0.1, inplace=True)

    return cost


if __name__ == '__main__':
    print("Test Correlation...")
    import time

    iters = 50
    scale = 8
    B, C, H, W = 1, 40, 384, 1248
    device = 'cuda:0'

    prev = torch.randn(B, C, H//scale, W//scale, device=device)
    curr = torch.randn(B, C, H//scale, W//scale, device=device)

    cost = correlation(prev, curr, patch_size=9, kernel_size=1)
    print('cost with shape: ', cost.shape)

    start_time = time.time()

    for i in range(iters):
        with torch.no_grad():
            correlation(prev, curr, patch_size=21, kernel_size=1)

            torch.cuda.synchronize(device)
    end_time = time.time()
    avg_time = (end_time - start_time) / iters


    print('{} reference forward once takes {:.4f}ms, i.e. {:.2f}fps'.format('Correlation', avg_time * 1000, (1 / avg_time)))

    print("Done!")

"""
Correlation2d: at scale=16, patch_size=21, reference forward once takes 0.6607ms, i.e. 1513.52fps 
"""
