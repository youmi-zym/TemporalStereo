import numpy as np
import cv2
from .load_disparity import load_eth3d_disp


def read_eth3d_intrinsic(intrinsic_fn):
    data = {}
    with open(intrinsic_fn, 'r') as fp:
        # 0 PINHOLE 941 490 542.019 542.019 541.922 255.202
        lines = fp.readlines()
        values = lines[-1][10:].rstrip().split(' ')
        h, w = values[1], values[0]
        resolution = (h, w)
        camera = '02'
        key = 'K_cam{}'.format(camera)
        inv_key = 'inv_K_cam{}'.format(int(camera))
        matrix = np.array([float(values[i]) for i in range(len(values))])
        K = np.eye(4)
        K[0, 0] = matrix[2]
        K[1, 1] = matrix[3]
        K[0, 2] = matrix[4]
        K[1, 2] = matrix[5]
        item = {
            key: K,
            inv_key: np.linalg.pinv(K)
        }
        data['{}'.format(camera)] = item

    return data, resolution


def read_eth3d_pfm_disparity(disp_fn, K=np.array([[541.764,        0,                  553.869,        0],
                                                  [0,              541.764,            232.396,        0],
                                                  [0,              0,                  1,              0],
                                                  [0,              0,                  0,              1]])):

    disp = load_eth3d_disp(disp_fn)
    # uint16
    valid_mask = disp > 0

    f = K[0, 0]
    b = 0.595499 # meter

    depth = b * f / (disp + 1e-12)
    depth = depth * valid_mask

    return depth, disp
