import torch
import warnings

from .flow_pixel_error import flow_calc_error

def do_flow_evaluation(est_flow, gt_flow, lb=0.0, ub=400, sparse=False):
    """
    Do pixel error evaluation. (See KITTI evaluation protocols for details.)
    Args:
        est_flow:                       (Tensor), estimated flow map
                                        [..., 2, Height, Width] layout
        gt_flow:                        (Tensor), ground truth flow map
                                        [..., 2, Height, Width] layout
        lb:                             (scalar), the lower bound of disparity you want to mask out
        ub:                             (scalar), the upper bound of disparity you want to mask out
        sparse:                         (bool), whether the given flow is sparse, default False
    Returns:
        error_dict (dict): the error of 1px, 2px, 3px, 5px, in percent,
            range [0,100] and average error epe
    """
    error_dict = {}
    if est_flow is None:
        warnings.warn('Estimated flow map is None')
        return error_dict
    if gt_flow is None:
        warnings.warn('Reference ground truth flow map is None')
        return error_dict

    if torch.is_tensor(est_flow):
        est_flow = est_flow.clone().cpu()

    if torch.is_tensor(gt_flow):
        gt_flow = gt_flow.clone().cpu()

    error_dict = flow_calc_error(est_flow, gt_flow, sparse=sparse)

    return error_dict
