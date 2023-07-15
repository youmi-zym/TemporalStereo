import warnings

import torch

from architecture.modeling.layers import inverse_warp
from .pixel_error import calc_error


def do_evaluation(est_disp, gt_disp, lb, ub):
    """
    Do pixel error evaluation. (See KITTI evaluation protocols for details.)
    Args:
        est_disp:                       (Tensor), estimated disparity map,
                                        [..., Height, Width] layout
        gt_disp:                        (Tensor), ground truth disparity map
                                        [..., Height, Width] layout
        lb:                             (scalar), the lower bound of disparity you want to mask out
        ub:                             (scalar), the upper bound of disparity you want to mask out
    Returns:
        error_dict:                     (dict), the error of 1px, 2px, 3px, 5px, in percent,
                                                range [0,100] and average error epe
    """
    error_dict = {}
    if est_disp is None:
        warnings.warn('Estimated disparity map is None')
        return error_dict
    if gt_disp is None:
        warnings.warn('Reference ground truth disparity map is None')
        return error_dict

    if torch.is_tensor(est_disp):
        est_disp = est_disp.clone().cpu()

    if torch.is_tensor(gt_disp):
        gt_disp = gt_disp.clone().cpu()

    assert est_disp.shape == gt_disp.shape, "Estimated Disparity map with shape: {}, but GroundTruth Disparity map" \
                                            " with shape: {}".format(est_disp.shape, gt_disp.shape)

    error_dict = calc_error(est_disp, gt_disp, lb=lb, ub=ub)

    return error_dict


def do_occlusion_evaluation(est_disp, ref_gt_disp, target_gt_disp, lb, ub):
    """
    Do occlusoin evaluation.
    Args:
        est_disp:                       (Tensor), estimated disparity map
                                        [BatchSize, 1, Height, Width] layout
        ref_gt_disp:                    (Tensor),   reference(left) ground truth disparity map
                                        [BatchSize, 1, Height, Width] layout
        target_gt_disp:                 (Tensor), target(right) ground truth disparity map,
                                        [BatchSize, 1, Height, Width]  layout
        lb:                             (scalar): the lower bound of disparity you want to mask out
        ub:                             (scalar): the upper bound of disparity you want to mask out
    Returns:
    """
    error_dict = {}
    if est_disp is None:
        warnings.warn('Estimated disparity map is None, expected given')
        return error_dict
    if ref_gt_disp is None:
        warnings.warn('Reference ground truth disparity map is None, expected given')
        return error_dict
    if target_gt_disp is None:
        warnings.warn('Target ground truth disparity map is None, expected given')
        return error_dict

    if torch.is_tensor(est_disp):
        est_disp = est_disp.clone().cpu()
    if torch.is_tensor(ref_gt_disp):
        ref_gt_disp = ref_gt_disp.clone().cpu()
    if torch.is_tensor(target_gt_disp):
        target_gt_disp = target_gt_disp.clone().cpu()

    assert est_disp.shape == ref_gt_disp.shape and target_gt_disp.shape == ref_gt_disp.shape, "{}, {}, {}".format(
                                                        est_disp.shape, ref_gt_disp.shape, target_gt_disp.shape)

    warp_ref_gt_disp = inverse_warp(target_gt_disp.clone(), -ref_gt_disp.clone(), mode='disparity')
    theta = 1.0
    eps = 1e-6
    occlusion = (
            (torch.abs(warp_ref_gt_disp.clone() - ref_gt_disp.clone()) > theta) |
            (torch.abs(warp_ref_gt_disp.clone()) < eps)
    ).prod(dim=1, keepdim=True).type_as(ref_gt_disp)
    occlusion = occlusion.clamp(0, 1)

    occlusion_error_dict = calc_error(
        est_disp.clone() * occlusion.clone(),
        ref_gt_disp.clone() * occlusion.clone(),
        lb=lb, ub=ub
    )
    for key in occlusion_error_dict.keys():
        error_dict['occ_' + key] = occlusion_error_dict[key]

    not_occlusion = 1.0 - occlusion
    not_occlusion_error_dict = calc_error(
        est_disp.clone() * not_occlusion.clone(),
        ref_gt_disp.clone() * not_occlusion.clone(),
        lb=lb, ub=ub
    )
    for key in not_occlusion_error_dict.keys():
        error_dict['noc_' + key] = not_occlusion_error_dict[key]

    return error_dict