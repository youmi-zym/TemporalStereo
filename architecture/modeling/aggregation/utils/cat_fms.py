import torch

from architecture.modeling.layers import inverse_warp_3d

def cat_fms(reference_fm, target_fm, disp_sample):
    """
    perform concatenation between left and rith image feature to construct 4D cost volume

    Args:
        reference_fm:                   (Tensor), the feature map of reference image, often left image
                                        [BatchSize, C, H, W]
        target_fm:                      (Tensor), the feature map of target image, often right image
                                        [BatchSize, C, H, W]
        disp_sample:                    (Tensor), the disparity samples/candidates for feature concatenation or matching
                                        [BatchSize, NumSamples, H, W]

    Returns:
        concat_fm:                      (Tensor), the concatenated feature map
                                        [BatchSize, 2C, NumSamples, H, W]
    """
    B, C, H, W = reference_fm.shape

    # the number of disparity samples
    D = disp_sample.shape[1]

    # expand D dimension
    concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
    concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

    # shift target feature according to disparity samples
    concat_target_fm = inverse_warp_3d(concat_target_fm, -disp_sample, padding_mode='zeros')

    # [B, 2C, D, H, W)
    concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

    return concat_fm


if __name__ == '__main__':
    """
    GPU: GTX3090, CUDA:11.0, Torch:1.7.1
    CAT_FMS reference forward once takes 5.3421ms, i.e. 187.19fps
    """
    print("Feature Concatenation Test...")
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


    cost = cat_fms(left, right, disp_samples)
    print('Cost in shape: ', cost.shape)
    idx = 0
    for i in range(-2, 3, 1):
        print('Disparity {}:\n {}'.format(i, cost[:, :, idx, ]))
        idx += 1

    for i in range(cost.shape[1]):
        print('Channel {}:\n {}'.format(i, cost[:, i, ]))

    # -------------------------------------- Time Test-------------------------------------- #

    C, H, W = 32, 384//4, 1248//4  # size in KITTI
    device = torch.device('cuda:0')
    left = torch.rand(1, C, H, W).to(device)
    right = torch.rand(1, C, H, W).to(device)

    max_disp = 192//4
    disp_samples = torch.linspace(0, max_disp-1, max_disp).repeat(1, H, W, 1). \
        permute(0, 3, 1, 2).contiguous().to(device)

    avg_time = timeTestTemplate(cat_fms, left, right, disp_samples, iters=50, device=torch.device('cuda:0'))

    print('{} reference forward once takes {:.4f}ms, i.e. {:.2f}fps'.format('CAT_FMS', avg_time * 1000, (1 / avg_time)))
    

