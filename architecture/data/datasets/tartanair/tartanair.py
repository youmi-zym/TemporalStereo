import numpy as np
import os

import torch

from architecture.data.utils import read_tartanair_flow, read_tartanair_extrinsic, read_tartanair_depth

from .base import TARTANAIRStereoDatasetBase

class TARTANAIRStereoDataset(TARTANAIRStereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(TARTANAIRStereoDataset, self).__init__(annFile, root, height, width, frame_idxs,
                                                     is_train, use_common_intrinsics, do_same_lr_transform,
                                                     mean, std)

    def Loader(self, image_path):
        image_path = os.path.join(self.root, image_path)
        img = self.img_loader(image_path)
        return img

    def depthLoader(self, depth_path, K=None):
        depth_path = os.path.join(self.root, depth_path)
        depth, disp = read_tartanair_depth(depth_path, K)

        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        disp = torch.from_numpy(disp.astype(np.float32)).unsqueeze(0)

        return depth, disp

    def flowLoader(self, flow_path):
        flow_path = os.path.join(self.root, flow_path)
        flow = read_tartanair_flow(flow_path)

        flow = torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1).contiguous()

        return flow

    def extrinsicLoader(self, extrinsic_path):
        # the transformation matrix is from world original point to current frame
        left_extrinsic_path = os.path.join(self.root, extrinsic_path, 'pose_left.txt')
        left_extrinsics = read_tartanair_extrinsic(left_extrinsic_path, 'left')
        right_extrinsic_path = os.path.join(self.root, extrinsic_path, 'pose_right.txt')
        right_extrinsics = read_tartanair_extrinsic(right_extrinsic_path, 'right')
        extrinsics = {}
        extrinsics.update(left_extrinsics)
        extrinsics.update(right_extrinsics)

        return extrinsics

    def getExtrinsic(self, extrinsics, image_path):
        # 'hospital/Easy/P000/image_left/000021_left.png'
        img_id = int(image_path.split('/')[-1].split('.')[0].split('_')[0])
        # the transformation matrix is from world original point to current frame
        # 4x4
        left_T = torch.from_numpy(extrinsics['Frame{}:0'.format(img_id)]['T_cam0'])
        left_inv_T = torch.from_numpy(extrinsics['Frame{}:0'.format(img_id)]['inv_T_cam0'])
        right_T = torch.from_numpy(extrinsics['Frame{}:1'.format(img_id)]['T_cam1'])
        right_inv_T = torch.from_numpy(extrinsics['Frame{}:1'.format(img_id)]['inv_T_cam1'])

        return left_T, left_inv_T, right_T, right_inv_T

    def intrinsicLoader(self, intrinsic_path):
        norm_K = self.K.copy()
        full_K = np.eye(4)
        resolution = self.full_resolution
        h, w = resolution
        full_K[0, :] = norm_K[0, :] * w
        full_K[1, :] = norm_K[1, :] * h

        return norm_K, full_K, resolution


