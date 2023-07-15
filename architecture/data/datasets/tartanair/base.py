import numpy as np
import os
from architecture.data.datasets.base import StereoDatasetBase

class TARTANAIRStereoDatasetBase(StereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        super(TARTANAIRStereoDatasetBase, self).__init__(annFile, root, height, width, frame_idxs,
                                                         is_train, use_common_intrinsics, do_same_lr_transform,
                                                         mean, std)

        self.K = np.array([[320.0/640.0,    0,              320.0/640.0,     0],
                           [0,              320.0/480.0,    240.0/480.0,     0],
                           [0,              0,              1,               0],
                           [0,              0,              0,               1]])
        # (h, w)
        self.full_resolution = (480, 640)

        self.baseline = 0.25

        self.with_depth_gt = True
        self.with_disp_gt = False
        self.with_flow_gt = False
        self.with_pose_gt = True
