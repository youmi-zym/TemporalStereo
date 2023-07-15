import numpy as np
import os
from architecture.data.datasets.base import StereoDatasetBase

class SceneFlowStereoDatasetBase(StereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(SceneFlowStereoDatasetBase, self).__init__(annFile, root, height, width, frame_idxs,
                                                         is_train, use_common_intrinsics, do_same_lr_transform,
                                                         mean, std)

        # https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#information
        # Most scenes use a virtual focal length of 35.0mm. For those scenes, the virtual camera intrinsics matrix is given by
        self.K = np.array([[1050.0/960,     0,              497.5/960,      0],
                           [0,              1050.0/540,     269.5/540,      0],
                           [0,              0,              1,              0],
                           [0,              0,              0,              1]])

        # Some scenes in the Driving subset use a virtual focal length of 15.0mm
        self.K15 = np.array([[450.0/960,    0,              497.5/960,      0],
                             [0,            450.0/540,      269.5/540,      0],
                             [0,            0,              1,              0],
                             [0,            0,              0,              1]])

        # (h, w)
        self.full_resolution = (540, 960)

        self.with_depth_gt = False
        self.with_disp_gt = True
        self.with_flow_gt = False
        self.with_pose_gt = True
