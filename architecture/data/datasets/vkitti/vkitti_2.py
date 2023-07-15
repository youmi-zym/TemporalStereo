import numpy as np
import os

import torch

from architecture.data.utils import read_vkitti_png_depth, read_vkitti_png_flow, read_vkitti_extrinsic, read_vkitti_intrinsic

from .base import VKITTIStereoDatasetBase

class VKITTI2StereoDataset(VKITTIStereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(VKITTI2StereoDataset, self).__init__(annFile, root, height, width, frame_idxs,
                                                   is_train, use_common_intrinsics, do_same_lr_transform,
                                                   mean, std)

    def Loader(self, image_path):
        image_path = os.path.join(self.root, image_path)
        img = self.img_loader(image_path)
        return img

    def depthLoader(self, depth_path, K=None):
        if self.use_common_intrinsics:
            K = np.eye(4)
            K[0, :] = self.K[0, :] * self.full_resolution[1]
            K[1, :] = self.K[1, :] * self.full_resolution[0]
        else:
            # 'Scene01/15-deg-left/frames/depth/Camera_0/depth_00001.png'
            scene, variation, _, _, camera, depth_name = depth_path.split('/')
            intrinsic_path = os.path.join(self.root, scene, variation, 'intrinsic.txt')
            intrinsics = read_vkitti_intrinsic(intrinsic_path)
            frame_idx = int(depth_name.split('.')[0].split('_')[-1])
            camera_idx = int(camera.split('_')[-1])
            K = intrinsics['Frame{}:{}'.format(frame_idx, camera_idx)]['K_cam{}'.format(camera_idx)]

        depth_path = os.path.join(self.root, depth_path)
        depth, disp = read_vkitti_png_depth(depth_path, K)

        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        disp = torch.from_numpy(disp.astype(np.float32)).unsqueeze(0)

        return depth, disp

    def flowLoader(self, flow_path):
        flow_path = os.path.join(self.root, flow_path)
        flow = read_vkitti_png_flow(flow_path)

        flow = torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1).contiguous()

        return flow

    def extrinsicLoader(self, extrinsic_path):
        # the transformation matrix is from world original point to current frame
        extrinsic_path = os.path.join(self.root, extrinsic_path)
        extrinsics = read_vkitti_extrinsic(extrinsic_path)

        return extrinsics

    def getExtrinsic(self, extrinsics, image_path):
        # 'Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00009.jpg'
        img_id = int(image_path.split('/')[-1].split('.')[0].split('_')[-1])
        # the transformation matrix is from world original point to current frame
        # 4x4
        left_T = torch.from_numpy(extrinsics['Frame{}:0'.format(img_id)]['T_cam0'])
        left_inv_T = torch.from_numpy(extrinsics['Frame{}:0'.format(img_id)]['inv_T_cam0'])
        right_T = torch.from_numpy(extrinsics['Frame{}:1'.format(img_id)]['T_cam1'])
        right_inv_T = torch.from_numpy(extrinsics['Frame{}:1'.format(img_id)]['inv_T_cam1'])

        return left_T, left_inv_T, right_T, right_inv_T

    # todo: vkiit2 intrinsic loader
    def intrinsicLoader(self, intrinsic_path):
        return self.K.copy(), self.full_resolution


