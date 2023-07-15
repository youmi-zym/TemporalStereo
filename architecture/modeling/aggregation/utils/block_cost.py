import torch
import torch.nn.functional as F

from architecture.modeling.layers import inverse_warp_3d

def groupwise_correlation(fea1, fea2):
    B, C, D, H, W = fea1.shape
    channels_per_group = 8
    assert C % channels_per_group == 0
    num_groups = C // channels_per_group
    cost = -torch.pow((fea1 - fea2), 2.0).view([B, num_groups, channels_per_group, D, H, W]).sum(dim=2)
    assert cost.shape == (B, num_groups, D, H, W)
    return cost


def block_cost(reference_fm, target_fm, disp_sample, block_cost_scale=3):
    """
    perform concatenation and groupwise correlation between left and rith image feature to construct 4D cost volume

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

    if isinstance(disp_sample, int):
        max_disp = disp_sample
        # [b, c, h, max_disp-1+w]
        padded_target_fm = F.pad(target_fm, pad=(max_disp-1, 0, 0, 0), mode='constant', value=0.0)
        unfolded_target_fm = F.unfold(padded_target_fm, kernel_size=(1, max_disp), dilation=(1, 1), padding=(0, 0), stride=(1, 1))
        unfolded_target_fm = unfolded_target_fm.reshape(B, C, max_disp, H, W)
        # [max_disp-1, ..., 2, 1, 0] -> [0, 1, 2, ..., max_disp-1]
        target_fm = torch.flip(unfolded_target_fm, dims=[2, ])

        reference_fm = reference_fm.reshape(B, C, 1, H, W).repeat(1, 1, max_disp, 1, 1)

        cost = -(reference_fm - target_fm) ** 2

    else:
        # the number of disparity samples
        D = disp_sample.shape[1]

        # expand D dimension
        reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
        target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)
        #
        # # shift target feature according to disparity samples
        target_fm = inverse_warp_3d(target_fm, -disp_sample, padding_mode='zeros')

        cost = torch.cat([reference_fm, target_fm], dim=1)


    # [B, C, D, H, W)
    B, C, D, H, W = reference_fm.shape

    costs = [cost, ]
    block_cost_scale = int(block_cost_scale)
    for s in range(block_cost_scale):
        sD, sH, sW = 1, min(2**s, H), min(2**s, W)
        local_reference_fm = F.avg_pool3d(reference_fm, kernel_size=(sD, sH, sW), stride=(sD, sH, sW))
        local_target_fm = F.avg_pool3d(target_fm, kernel_size=(sD, sH, sW), stride=(sD, sH, sW))

        cost = groupwise_correlation(local_reference_fm, local_target_fm)

        # [B, C//8, D, H, W]
        cost = F.interpolate(cost, size=(D, H, W), mode='trilinear', align_corners=True)

        cost = cost.reshape(B, C//8, D, H, W).contiguous()

        costs.append(cost)

    # [B, 2C + C//8*local_scale, D, H, W]
    cost = torch.cat(costs, dim=1)

    return cost


if __name__ == '__main__':
    """
    GPU: GTX3090, CUDA:11.0, Torch:1.7.1
    SPARSE_CAT_FMS reference forward once takes 3.1339ms, i.e. 319.09fps at 1/4 resolution, 2.4887ms, i.e. 401.82fps with 2 scale
    SPARSE_CAT_FMS reference forward once takes 1.0396ms, i.e. 961.90fps at 1/8 resolution
    
    BLOCK_COST reference forward once takes 1.7147ms, i.e. 583.18fps at 1/4 resolution, C=48, disp_samples=4
    """
    print("Feature Concatenation Test...")
    from architecture.utils import timeTestTemplate

    # -------------------------------------- Time Test-------------------------------------- #

    scale = 16
    C, H, W = 192, 384//scale, 1248//scale  # size in KITTI
    device = torch.device('cuda:0')
    left = torch.rand(1, C, H, W).to(device)
    right = torch.rand(1, C, H, W).to(device)

    disp_samples = (torch.randn(12) * W).repeat(1, H, W, 1). \
        permute(0, 3, 1, 2).contiguous().to(device)
    # disp_samples = 12

    avg_time = timeTestTemplate(block_cost, left, right, disp_samples, iters=1000, device=torch.device('cuda:0'))

    print('{} reference forward once takes {:.4f}ms, i.e. {:.2f}fps'.format('BLOCK_COST', avg_time * 1000, (1 / avg_time)))
    

