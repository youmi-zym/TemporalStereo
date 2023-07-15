import numpy as np
import os

import torch

from architecture.data.utils import read_sceneflow_pfm_disparity, read_sceneflow_pfm_flow, read_sceneflow_extrinsic

from .base import SceneFlowStereoDatasetBase

class SceneFlowStereoDataset(SceneFlowStereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(SceneFlowStereoDataset, self).__init__(annFile, root, height, width, frame_idxs,
                                                     is_train, use_common_intrinsics, do_same_lr_transform,
                                                     mean, std)

    def Loader(self, image_path):
        image_path = os.path.join(self.root, image_path)
        try:
            img = self.img_loader(image_path)
            return img
        except:
            print("Image Reading Error: {}".format(image_path))
            exit(-1)

    def dispLoader(self, disp_path, K=None):
        disp_path = os.path.join(self.root, disp_path)
        depth, disp = read_sceneflow_pfm_disparity(disp_path, K)

        depth = torch.from_numpy(depth.copy().astype(np.float32)).unsqueeze(0)
        disp = torch.from_numpy(disp.copy().astype(np.float32)).unsqueeze(0)

        return depth, disp

    def flowLoader(self, flow_path):
        flow_path = os.path.join(self.root, flow_path)
        flow = read_sceneflow_pfm_flow(flow_path)

        flow = torch.from_numpy(flow.copy().astype(np.float32)).unsqueeze(0)

        return flow

    def intrinsicLoader(self, intrinsic_path):
        if '15mm' in intrinsic_path:
            norm_K = self.K15.copy()
        else:
            norm_K = self.K.copy()

        full_K = np.eye(4)
        resolution = self.full_resolution
        h, w = resolution
        full_K[0, :] = norm_K[0, :] * w
        full_K[1, :] = norm_K[1, :] * h

        return norm_K, full_K, resolution

    def extrinsicLoader(self, extrinsic_path):
        # the transformation matrix is from world original point to current frame
        extrinsic_path = os.path.join(self.root, extrinsic_path)
        extrinsics = read_sceneflow_extrinsic(extrinsic_path)
        return extrinsics

    def getExtrinsic(self, extrinsics, image_path):
        # 'SceneFlow/FlyingThings3D/frames_cleanpass/TRAIN/A/0000/left/0006.png'
        img_id = int(image_path.split('/')[-1].split('.')[0])
        # the transformation matrix is from world original point to current frame
        # 4x4
        try:
            left_T = torch.from_numpy(extrinsics['Frame{}:0'.format(img_id)]['T_cam0'])
            left_inv_T = torch.from_numpy(extrinsics['Frame{}:0'.format(img_id)]['inv_T_cam0'])
            right_T = torch.from_numpy(extrinsics['Frame{}:1'.format(img_id)]['T_cam1'])
            right_inv_T = torch.from_numpy(extrinsics['Frame{}:1'.format(img_id)]['inv_T_cam1'])
        except:
            # print("There is no extrinsic parameter for image: {}, set to identical matrix!".format(image_path))
            left_T = torch.eye(4, 4)
            left_inv_T = torch.eye(4, 4)
            right_T = torch.eye(4, 4)
            right_inv_T = torch.eye(4, 4)

        return left_T, left_inv_T, right_T, right_inv_T
