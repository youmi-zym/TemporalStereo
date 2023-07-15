import numpy as np
import os

import torch

from architecture.data.utils import read_kitti_intrinsic, read_kitti_extrinsic, read_kitti_png_disparity

from .base import KITTIStereoDatasetBase

class KITTI2015StereoDataset(KITTIStereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(KITTI2015StereoDataset, self).__init__(annFile, root, height, width, frame_idxs,
                                                     is_train, use_common_intrinsics, do_same_lr_transform,
                                                     mean, std)

    def Loader(self, image_path):
        image_path = os.path.join(self.root, image_path)
        img = self.img_loader(image_path)
        return img

    def dispLoader(self, disp_path, K=None):
        disp_path = os.path.join(self.root, disp_path)
        depth, disp = read_kitti_png_disparity(disp_path, K)

        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        disp = torch.from_numpy(disp.astype(np.float32)).unsqueeze(0)

        return depth, disp

    def extrinsicLoader(self, extrinsic_path):
        # the transformation matrix is from world original point to current frame
        extrinsic_path = os.path.join(self.root, extrinsic_path)
        extrinsics = read_kitti_extrinsic(extrinsic_path)

        return extrinsics

    def getExtrinsic(self, extrinsics, image_path):
        # 'testing/sequences/000000/image_2/000000_10.png'
        img_id = int(image_path.split('/')[-1].split('.')[0].split('_')[-1])
        # the transformation matrix is from world original point to current frame
        # 4x4
        left_T = torch.from_numpy(extrinsics['Frame{:02d}:02'.format(img_id)]['T_cam02'])
        left_inv_T = torch.from_numpy(extrinsics['Frame{:02d}:02'.format(img_id)]['inv_T_cam02'])
        pose = extrinsics.get('Frame{:02d}:03', None)
        if pose is not None:
            right_T = pose['T_cam03']
            right_inv_T = pose['inv_T_cam03']
        else:
            right_T = torch.eye(4)
            right_inv_T = torch.eye(4)

        return left_T, left_inv_T, right_T, right_inv_T

    def intrinsicLoader(self, intrinsic_path):
        intrinsic_path = os.path.join(self.root, intrinsic_path)
        K, resolution = read_kitti_intrinsic(intrinsic_path)
        K = K['02']['K_cam02']
        full_K = K.copy()
        h, w = resolution
        norm_K = np.eye(4)
        norm_K[0, :] = K[0, :] / w
        norm_K[1, :] = K[1, :] / h
        return norm_K.copy(), full_K, resolution


