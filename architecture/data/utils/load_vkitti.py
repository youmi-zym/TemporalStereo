import numpy as np
import cv2


def read_vkitti_intrinsic(intrinsic_fn):
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


def read_vkitti_extrinsic(extrinsic_fn):
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


def read_vkitti_png_depth(depth_fn, K=np.array([[725.0087,       0,              620.5,     0],
                                                [0,              725.0087,       187,       0],
                                                [0,              0,              1,         0],
                                                [0,              0,              0,         1]])):

    depth = cv2.imread(depth_fn, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    # [0, 655.35] meter
    depth = depth / 100.0

    f = K[0, 0]
    b = 0.532725 # meter

    disp = b * f / depth

    return depth, disp

def read_vkitti_png_flow(flow_fn):
    """Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"""
    # read png to bgr in 16 bit unsigned short

    bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0 # or another value (e.g., np.nan)

    return out_flow

