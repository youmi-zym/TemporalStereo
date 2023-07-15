import numpy as np
import cv2


def read_drivingstereo_intrinsic(intrinsic_fn):
    lineid = 0
    data = {}
    with open(intrinsic_fn, 'r') as fp:
        for line in fp.readlines():
            values = line.rstrip().split(' ')
            if lineid == 0:
                lineid += 1
                continue
            frame = int(values[0])
            camera = int(values[1])
            key = 'K_cam{}'.format(camera)
            inv_key = 'inv_K_cam{}'.format(int(camera))
            matrix = np.array([float(values[i]) for i in range(2, len(values))])
            K = np.eye(4)
            K[0, 0] = matrix[0]
            K[1, 1] = matrix[1]
            K[0, 2] = matrix[2]
            K[1, 2] = matrix[3]
            item = {
                key: K,
                inv_key: np.linalg.pinv(K)
            }
            data['Frame{}:{}'.format(frame, camera)] = item
            lineid += 1

    return data


def read_drivingstereo_extrinsic(extrinsic_fn):
    lineid = 0
    data = {}
    with open(extrinsic_fn, 'r') as fp:
        for line in fp.readlines():
            values = line.rstrip().split(' ')
            if lineid == 0:
                lineid += 1
                continue
            frame = int(values[0])
            camera = int(values[1])
            key = 'T_cam{}'.format(int(camera))
            inv_key = 'inv_T_cam{}'.format(int(camera))
            matrix = np.array([float(values[i]) for i in range(2, len(values))])
            matrix = matrix.reshape(4, 4)
            item = {
                key: matrix,
                inv_key: np.linalg.pinv(matrix),
            }
            data['Frame{}:{}'.format(frame, camera)] = item
            lineid += 1

    return data


def read_drivingstereo_png_depth(depth_fn, K=np.array([[725.0087,       0,              620.5,     0],
                                                [0,              725.0087,       187,       0],
                                                [0,              0,              1,         0],
                                                [0,              0,              0,         1]]), div_factor=256.0):

    depth = cv2.imread(depth_fn, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    # The disparity value and depth value for each pixel can be computed by
    # converting the uint16 value to float and dividing it by 256.
    # The zero values indicate the invalid pixels.
    # Different from half-resulution disparity maps,
    # the disparity value for each pixel in the full-resolution map
    # is computed by converting the uint16 value to float and dividing it by 128.
    # uint16, full div 128, hafl div 256
    valid = depth > 0
    depth = (depth * valid) / div_factor

    f = K[0, 0]
    b = 0.5443450 # meter

    disp = b * f / depth

    return depth, disp

def read_drivingstereo_png_flow(flow_fn):
    raise NotImplementedError("DrivingStereo has no ground truth for optical flow!!!")

