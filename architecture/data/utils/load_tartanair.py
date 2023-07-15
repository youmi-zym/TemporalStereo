import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def read_tartantic_intrinsic():
    K = np.eye(4)
    K[0, 0] = 320.0
    K[1, 1] = 320.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0

    return K


def read_tartanair_extrinsic(extrinsic_fn, side='left'):
    data = {}
    camera_id = {'left': 0, 'right': 1}
    with open(extrinsic_fn, 'r') as fp:
        lines = fp.readlines()
    # poses = np.loadtxt(extrinsic_fn)
    # for lineid, pose in enumerate(poses):
    for lineid, line in enumerate(lines):
        frame = int(lineid)
        camera = int(camera_id[side])
        key = 'T_cam{}'.format(int(camera))
        inv_key = 'inv_T_cam{}'.format(int(camera))
        values = line.rstrip().split(' ')
        assert len(values) == 7, 'Pose must be quaterion format -- 7 params, but {} got'.format(len(values))
        pose = np.array([float(values[i]) for i in range(len(values))])
        tx, ty, tz, qx, qy, qz, qw = pose
        R = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
        t = np.array([tx, ty, tz])
        matrix = np.eye(4)
        matrix[:3, :3] = R.transpose()
        matrix[:3, 3] = -R.transpose().dot(t)
        # ned(z-axis down) to z-axis forward
        m_correct = np.zeros_like(matrix)
        m_correct[0, 1] = 1
        m_correct[1, 2] = 1
        m_correct[2, 0] = 1
        m_correct[3, 3] = 1
        matrix = np.matmul(m_correct, matrix)

        item = {
            key: matrix,
            inv_key: np.linalg.pinv(matrix),
        }
        data['Frame{}:{}'.format(frame, camera)] = item
        lineid += 1

    return data


def read_tartanair_depth(depth_fn, K=np.array([[320.0,      0,      320.0,      0],
                                                [0,         320,    240.0,      0],
                                                [0,         0,      1,          0],
                                                [0,         0,      0,          1]])):

    if '.npy' in depth_fn:
        depth = np.load(depth_fn)
    elif '.png' in depth_fn:
        depth = cv2.imread(depth_fn, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        # [0, 655.35] meter
        depth = depth / 100.0
    else:
        raise TypeError('only support png and npy format, invalid type found: {}'.format(depth_fn))

    f = K[0, 0]
    b = 0.25 # meter

    disp = b * f / (depth + 1e-5)

    return depth, disp

def read_tartanair_flow(flow_fn):
    """Convert to (h, w, 2) (flow_x, flow_y) float32 array"""

    out_flow = np.load(flow_fn)

    return out_flow
