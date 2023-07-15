import cv2
import numpy as np
from .load_flow import load_flying_things_flow
from .load_disparity import load_scene_flow_disp


def read_sceneflow_extrinsic(extrinsic_fn):
    data = {}
    with open(extrinsic_fn, 'r') as fp:
        lines = fp.readlines()
        item_num = len(lines) // 4
        for i in range(item_num):
            frame_info = lines[i*4+0]
            frame = int(frame_info.rstrip().split(' ')[-1])

            # read left camera extrinsic
            left_extrinisc = lines[i*4+1]
            left_values = left_extrinisc.rstrip().split(' ')
            camera = 0
            key = 'T_cam{}'.format(int(camera))
            inv_key = 'inv_T_cam{}'.format(int(camera))
            matrix = np.array([float(left_values[i]) for i in range(1, len(left_values))])
            matrix = matrix.reshape(4, 4)
            item = {
                key: matrix,
                inv_key: np.linalg.pinv(matrix),
            }
            data['Frame{}:{}'.format(frame, camera)] = item

            # read right camera extrinsic
            right_extrinisc = lines[i*4+2]
            right_values = right_extrinisc.rstrip().split(' ')
            camera = 1
            key = 'T_cam{}'.format(int(camera))
            inv_key = 'inv_T_cam{}'.format(int(camera))
            matrix = np.array([float(right_values[i]) for i in range(1, len(right_values))])
            matrix = matrix.reshape(4, 4)
            item = {
                key: matrix,
                inv_key: np.linalg.pinv(matrix),
            }
            data['Frame{}:{}'.format(frame, camera)] = item

    return data



def read_sceneflow_pfm_disparity(disp_fn, K):
    disp  = load_scene_flow_disp(disp_fn)
    h, w = disp.shape
    disp = np.nan_to_num(disp, nan=0.0)
    disp[disp > w] = 0
    disp[disp < 0] = 0

    f = K[0, 0]
    # https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#information, No.6
    b = 1.0 # meter
    eps = 1e-8

    depth = b * f / (disp + eps)

    return depth, disp

def read_sceneflow_pfm_flow(flow_fn):
    """Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"""
    # read png to bgr in 16 bit unsigned short

    out_flow = load_flying_things_flow(flow_fn)

    return out_flow
