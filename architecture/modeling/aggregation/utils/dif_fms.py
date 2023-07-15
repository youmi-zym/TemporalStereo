import torch

from architecture.modeling.layers import inverse_warp_3d

def dif_fms(reference_fm, target_fm, disp_sample):
    """
    perform substraction(i.e., compare difference) between left and rith image feature to construct 4D cost volume

    Args:
        reference_fm:                   (Tensor), the feature map of reference image, often left image
                                        [BatchSize, C, H, W]
        target_fm:                      (Tensor), the feature map of target image, often right image
                                        [BatchSize, C, H, W]
        disp_sample:                    (Tensor), the disparity samples/candidates for feature concatenation or matching
                                        [BatchSize, NumSamples, H, W]

    Returns:
        dif_fm:                         (Tensor), the substracted feature map
                                        [BatchSize, C, NumSamples, H, W]
    """
    B, C, H, W = reference_fm.shape

    # the number of disparity samples
    D = disp_sample.shape[1]

    # expand D dimension
    dif_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
    dif_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

    # shift target feature according to disparity samples
    dif_target_fm = inverse_warp_3d(dif_target_fm, -disp_sample, padding_mode='zeros')

    # [B, C, D, H, W)
    dif_fm = torch.abs(dif_reference_fm - dif_target_fm)

    # fill the outliers with max cost
    max_dif = dif_fm.max()
    ones = torch.ones_like(dif_fm)
    no_empty_mask = (dif_target_fm > 0).float()
    # [B, C, D, H, W)
    dif_fm = (dif_fm * no_empty_mask) + (1 - no_empty_mask) * ones * max_dif


    return dif_fm


if __name__ == '__main__':
    """
    GPU: GTX3090, CUDA:11.0, Torch:1.7.1
    DIF_FMS reference forward once takes 8.3691ms, i.e. 119.49fps
    """
    print("Feature Substraction Test...")
    from architecture.utils import timeTestTemplate

    # -------------------------------------- Value Test-------------------------------------- #
    H, W = 3, 4
    device = torch.device('cuda:0')
    left = torch.linspace(1, H * W, H * W).reshape(1, 1, H, W).to(device)
    right = torch.linspace(H * W + 1, H * W * 2, H * W).reshape(1, 1, H, W).to(device)
    print('left: \n ', left)
    print('right: \n ', right)

    disp_samples = torch.linspace(-2, 2, 5).repeat(1, H, W, 1). \
        permute(0, 3, 1, 2).contiguous().to(device)

    print("Disparity Samples/Candidates: \n", disp_samples)

    cost = dif_fms(left, right, disp_samples)
    print('Cost in shape: ', cost.shape)
    idx = 0
    for i in range(-2, 3, 1):
        print('Disparity {}:\n {}'.format(i, cost[:, :, idx, ]))
        idx += 1

    for i in range(cost.shape[1]):
        print('Channel {}:\n {}'.format(i, cost[:, i, ]))

    # -------------------------------------- Time Test-------------------------------------- #

    C, H, W = 32, 384 // 4, 1248 // 4  # size in KITTI
    device = torch.device('cuda:0')
    left = torch.rand(1, C, H, W).to(device)
    right = torch.rand(1, C, H, W).to(device)

    max_disp = 192 // 4
    disp_samples = torch.linspace(0, max_disp - 1, max_disp).repeat(1, H, W, 1). \
        permute(0, 3, 1, 2).contiguous().to(device)

    avg_time = timeTestTemplate(dif_fms, left, right, disp_samples)

    print('{} reference forward once takes {:.4f}ms, i.e. {:.2f}fps'.format('DIF_FMS', avg_time * 1000, (1 / avg_time)))


