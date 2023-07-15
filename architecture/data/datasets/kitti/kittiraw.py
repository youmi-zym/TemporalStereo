import numpy as np
import os

import torch

from architecture.data.utils.calibration.kitti_calib import load_calib, read_calib_file
from architecture.data.utils import read_kitti_intrinsic, read_kitti_extrinsic, read_kitti_png_disparity

from .base import KITTIStereoDatasetBase

class KITTIRAWStereoDataset(KITTIStereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(KITTIRAWStereoDataset, self).__init__(annFile, root, height, width, frame_idxs,
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
        lineid = 0
        data = {}
        extrinsic_path = os.path.join(self.root, extrinsic_path)
        with open(extrinsic_path, 'r') as fp:
            for line in fp.readlines():
                values = line.rstrip().split(' ')
                frame = '{:04d}'.format(lineid)
                camera = '02'
                key = 'T_cam{}'.format(camera)
                inv_key = 'inv_T_cam{}'.format(camera)
                matrix = np.array([float(values[i]) for i in range(len(values))])
                matrix = matrix.reshape(3, 4)
                matrix = np.concatenate((matrix, np.array([[0, 0, 0, 1]])), axis=0)
                item = {
                    # inverse pose
                    key: np.linalg.pinv(matrix),
                    inv_key: matrix,
                }
                data['Frame{}:{}'.format(frame, camera)] = item
                lineid += 1

        return data

    def getExtrinsic(self, extrinsics, image_path):
        # rawdata/2011_09_26/2011_09_26_drive_0095_sync/image_03/data/0000000001.png
        img_id = int(image_path.split('/')[-1].split('.')[0])
        # the transformation matrix is from world original point to current frame
        # 4x4
        left_T = torch.from_numpy(extrinsics['Frame{:04d}:02'.format(img_id)]['T_cam02'])
        left_inv_T = torch.from_numpy(extrinsics['Frame{:04d}:02'.format(img_id)]['inv_T_cam02'])
        pose = extrinsics.get('Frame{:04d}:03', None)
        if pose is not None:
            right_T = pose['T_cam03']
            right_inv_T = pose['inv_T_cam03']
        else:
            right_T = torch.eye(4)
            right_inv_T = torch.eye(4)

        return left_T, left_inv_T, right_T, right_inv_T

    def intrinsicLoader(self, intrinsic_path):
        intrinsic_path = os.path.join(self.root, intrinsic_path)
        data = read_calib_file(intrinsic_path)
        K = data['P_rect_02'].reshape(3, 4)[:3, :3]
        resolution = data['S_rect_02'][[1,0]]
        full_K = K.copy()
        h, w = resolution
        norm_K = np.eye(4)
        norm_K[0, :3] = K[0, :] / w
        norm_K[1, :3] = K[1, :] / h
        return norm_K.copy(), full_K, resolution


