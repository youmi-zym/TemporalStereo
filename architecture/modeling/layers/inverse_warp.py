import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_warp(img: torch.Tensor,
                 motion: torch.Tensor,
                 mode: str ='disparity',
                 K: torch.Tensor = None,
                 inv_K: torch.Tensor = None,
                 T_target_to_source: torch.Tensor = None,
                 interpolate_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 eps: float = 1e-7,
                 output_all: bool = False):
    """
    sample the image pixel value from source image and project to target image space,
    Args:
        img:                            (Tensor): the source image (where to sample pixels)
                                        [BatchSize, C, Height, Width]
        motion:                         (Tensor): disparity/depth/flow map of the target image
                                        [BatchSize, 1, Height, Width]
        mode:                           (str): which kind of warp to perform, including: ['disparity', 'depth', 'flow']
        K:                              (Optional, Tensor): instrincs of camera
                                        [BatchSize, 3, 3] or [BatchSize, 4, 4]
        inv_K:                          (Optional, Tensor): invserse instrincs of camera
                                        [BatchSize, 3, 3] or [BatchSize, 4, 4]
        T_target_to_source:             (Optional, Tensor): predicted transformation matrix from target image to source image frames
                                        [BatchSize, 4, 4]
        interpolate_mode:               (str): interpolate mode when grid sample, default is bilinear
        padding_mode:                   (str): padding mode when grid sample, default is zero padding
        eps:                            (float): eplison value to avoid divide 0, default 1e-7
        output_all:                     (bool): if output all result during warp, default False
    Returns:
        projected_img:                  (Tensor): source image warped to the target image
                                        [BatchSize, C, Height, Width]
        output:                         (Optional, Dict): such as optical flow, flow mask, triangular depth, src_pixel_coord and so on
    """
    B, C, H, W = motion.shape
    device = motion.device
    dtype = motion.dtype
    output = {}

    if mode == 'disparity':
        assert C == 1, "Disparity map must be 1 channel, but {} got!".format(C)
        # [B, 2, H, W]
        pixel_coord = mesh_grid(B, H, W, device, dtype)
        X = pixel_coord[:, 0, :, :] + motion[:, 0]
        Y = pixel_coord[:, 1, :, :]
    elif mode == 'flow':
        assert C == 2, "Optical flow map must be 2 channel, but {} got!".format(C)
        # [B, 2, H, W]
        pixel_coord = mesh_grid(B, H, W, device, dtype)
        X = pixel_coord[:, 0, :, :] + motion[:, 0]
        Y = pixel_coord[:, 1, :, :] + motion[:, 1]
    elif mode == 'depth':
        assert C == 1, "Disparity map must be 1 channel, but {} got!".format(C)
        outs = project_to_3d(motion, K, inv_K, T_target_to_source, eps)
        output.update(outs)
        src_pixel_coord = outs['src_pixel_coord']
        X = src_pixel_coord[:, 0, :, :]
        Y = src_pixel_coord[:, 1, :, :]

    else:
        raise TypeError("Inverse warp only support [disparity, flow, depth] mode, but {} got".format(mode))

    X_norm = 2 * X / (W-1) - 1
    Y_norm = 2 * Y / (H-1) - 1
    # [B, H, W, 2]
    pixel_coord_norm = torch.stack((X_norm, Y_norm), dim=3)

    projected_img = F.grid_sample(img, pixel_coord_norm, mode=interpolate_mode, padding_mode=padding_mode, align_corners=True)

    if output_all:
        return projected_img, output
    else:
        return projected_img


def mesh_grid(b, h, w, device, dtype=torch.float):
    """ construct pixel coordination in an image"""
    # [1, H, W]  copy 0-width for h times  : x coord
    x_range = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    # [1, H, W]  copy 0-height for w times : y coord
    y_range = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)

    # [b, 2, h, w]
    pixel_coord = torch.cat((x_range, y_range), dim=1)

    return pixel_coord

def project_to_3d(depth, K, inv_K=None, T_target_to_source:torch.Tensor=None, eps=1e-7):
    """
    project depth map to 3D space
    Args:
        depth:                          (Tensor): depth map(s), can be several depth maps concatenated at channel dimension
                                        [BatchSize, Channel, Height, Width]
        K:                              (Tensor): instrincs of camera
                                        [BatchSize, 3, 3] or [BatchSize, 4, 4]
        inv_K:                          (Optional, Tensor): invserse instrincs of camera
                                        [BatchSize, 3, 3] or [BatchSize, 4, 4]
        T_target_to_source:             (Optional, Tensor): predicted transformation matrix from target image to source image frames
                                        [BatchSize, 4, 4]
        eps:                            (float): eplison value to avoid divide 0, default 1e-7

    Returns: Dict including
        homo_points_3d:                 (Tensor): the homogeneous points after depth project to 3D space, [x, y, z, 1]
                                        [BatchSize, 4, Channel*Height*Width]
    if T_target_to_source provided:

        triangular_depth:               (Tensor): the depth map after the 3D points project to source camera
                                        [BatchSize, Channel, Height, Width]
        optical_flow:                   (Tensor): by 3D projection, the rigid flow can be got
                                        [BatchSize, Channel*2, Height, Width], to get the 2nd flow, index like [:, 2:4, :, :]
        flow_mask:                      (Tensor): the mask indicates which pixel's optical flow is valid
                                        [BatchSize, Channel, Height, Width]
    """

    # support C >=1, for C > 1, it means several depth maps are concatenated at channel dimension
    B, C, H, W = depth.size()
    device = depth.device
    dtype = depth.dtype
    output = {}

    # [B, 2, H, W]
    pixel_coord = mesh_grid(B, H, W, device, dtype)
    ones = torch.ones(B, 1, H, W, device=device, dtype=dtype)
    # [B, 3, H, W], homogeneous coordination of image pixel, [x, y, 1]
    homo_pixel_coord = torch.cat((pixel_coord, ones), dim=1).contiguous()

    # [B, 3, H*W] -> [B, 3, C*H*W]
    homo_pixel_coord = homo_pixel_coord.view(B, 3, -1).repeat(1, 1, C).contiguous()
    # [B, C*H*W] -> [B, 1, C*H*W]
    depth = depth.view(B, -1).unsqueeze(dim=1).contiguous()
    if inv_K is None:
        inv_K = torch.inverse(K[:, :3, :3])
    # [B, 3, C*H*W]
    points_3d = torch.matmul(inv_K[:, :3, :3], homo_pixel_coord) * depth
    ones = torch.ones(B, 1, C*H*W, device=device, dtype=dtype)
    # [B, 4, C*H*W], homogeneous coordiate, [x, y, z, 1]
    homo_points_3d = torch.cat((points_3d, ones), dim=1)
    output['homo_points_3d'] = homo_points_3d

    if T_target_to_source is not None:
        if K.shape[-1] == 3:
            new_K = torch.eye(4, device=device, dtype=dtype).unsqueeze(dim=0).repeat(B, 1, 1)
            new_K[:, :3, :3] = K[:, :3, :3]
            # [B, 3, 4]
            P = torch.matmul(new_K, T_target_to_source)[:, :3, :]
        else:
            # [B, 3, 4]
            P = torch.matmul(K, T_target_to_source)[:, :3, :]
        # [B, 3, C*H*W]
        src_points_3d = torch.matmul(P, homo_points_3d)

        # [B, C*H*W] -> [B, C, H, W], the depth map after 3D points projected to source camera
        triangular_depth = src_points_3d[:, -1, :].reshape(B, C, H, W).contiguous()
        output['triangular_depth'] = triangular_depth
        # [B, 2, C*H*W]
        src_pixel_coord = src_points_3d[:, :2, :] / (src_points_3d[:, 2:3, :] + eps)
        # [B, 2, C, H, W] -> [B, C, 2, H, W]
        src_pixel_coord = src_pixel_coord.reshape(B, 2, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # [B, C, 1, H, W]
        mask = (src_pixel_coord[:, :, 0:1] >=0) & (src_pixel_coord[:, :, 0:1] <= W-1) \
               & (src_pixel_coord[:, :, 1:2] >=0) & (src_pixel_coord[:, :, 1:2] <= H-1)

        # valid flow mask
        mask = mask.reshape(B, C, H, W).contiguous()
        output['flow_mask'] = mask
        # [B, C*2, H, W]
        src_pixel_coord = src_pixel_coord.reshape(B, C*2, H, W).contiguous()
        output['src_pixel_coord'] = src_pixel_coord
        # [B, C*2, H, W]
        optical_flow = src_pixel_coord - pixel_coord.repeat(1, C, 1, 1)
        output['optical_flow'] = optical_flow

    return output