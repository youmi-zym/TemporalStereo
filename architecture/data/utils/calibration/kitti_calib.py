import os
import numpy as np

# Extract calibration information from KITTI
# We use https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py as reference
# for the calibration and poses loading

def read_calib_file(filepath):
    """Read a calibration file and parse into a dictionary"""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def read_calib_from_video(cam2cam_filepath, velo2cam_filepath):
    """Read the cam_to_cam calibration and velo_to_cam calibration and parse into a dictionary"""
    data = {}
    cam2cam = read_calib_file(cam2cam_filepath)
    velo2cam = read_calib_file(velo2cam_filepath)

    # Transformation matrix from rotation matrix and translation vector, in 3x4 layout
    Tr_velo_to_cam = np.zeros((3, 4))
    Tr_velo_to_cam[:3, :3] = np.reshape(velo2cam['R'], [3, 3])
    Tr_velo_to_cam[:, 3] = velo2cam['T']

    data['Tr_velo_to_cam'] = Tr_velo_to_cam
    data.update(cam2cam)
    return data

def load_calib(calib_filepath, velo2cam_filepath=None):
    """
    For KITTI visual odometry:
        calibration stored in one file: P0, P1, P2, P3, Tr
        P0/P1/P2/P3 are the 3x4 projection matrices after rectification
        Tr transforms a point from velodyne coordinates into the left rectified camera coordinate system
    For KITTI raw data:
        calibration stored in two files(calib_cam_to_cam.txt, calib_velo_to_cam.txt)
        S_0x, K_0x, D_0x, T_0x, S_rect_0x, R_rect_0x, P_rect_0x -> for x in [0, 1, 2, 3]
    For KITTI 2015:
        calibration stored in two files(calib_cam_to_cam/{:0>6d}.txt, calib_velo_to_cam/{:0>6d.txt})
        contents are the same as KITTI raw data

    """
    from .utils import cart_to_homo, trans2homo4x4

    data = {}
    assert os.path.isfile(calib_filepath), calib_filepath
    if velo2cam_filepath is not None:
        cam2cam_filepath = calib_filepath
        assert os.path.isfile(velo2cam_filepath), velo2cam_filepath
        fileData = read_calib_from_video(cam2cam_filepath, velo2cam_filepath)
    else:
        fileData = read_calib_file(calib_filepath)

    # Create 3x4 projection matrices
    if 'P0' in fileData.keys(): # KITTI visual odometry
        P0, P1, P2, P3 = ['P{}'.format(x) for x in range(4)]
    elif 'P_rect_00' in fileData.keys(): # KITTI raw data, KITTI 2015
        P0, P1, P2, P3 = ['P_rect_0{}'.format(x) for x in range(4)]
    else:
        raise ValueError

    P_rect_00 = data['P_rect_00'] = np.reshape(fileData[P0], (3, 4))
    P_rect_10 = data['P_rect_10'] = np.reshape(fileData[P1], (3, 4))
    P_rect_20 = data['P_rect_20'] = np.reshape(fileData[P2], (3, 4))
    P_rect_30 = data['P_rect_30'] = np.reshape(fileData[P3], (3, 4))

    # Compute the camera intrinsics, in 3x3 layout
    K0 = data['K_cam0'] = P_rect_00[:3, :3]
    K1 = data['K_cam1'] = P_rect_10[:3, :3]
    K2 = data['K_cam2'] = P_rect_20[:3, :3]
    K3 = data['K_cam3'] = P_rect_30[:3, :3]

    # Compute the rectified extrinsic from cam0 to camN, in 4x4 layout
    # P_X0 = KX @ T_X0, so T_X0 = inv(KX) @ P_X0
    T0 = np.linalg.inv(trans2homo4x4(K0)) @ trans2homo4x4(P_rect_00)
    T1 = np.linalg.inv(trans2homo4x4(K1)) @ trans2homo4x4(P_rect_10)
    T2 = np.linalg.inv(trans2homo4x4(K2)) @ trans2homo4x4(P_rect_20)
    T3 = np.linalg.inv(trans2homo4x4(K3)) @ trans2homo4x4(P_rect_30)

    # Compute the velodyne to rectified camera coordinate transforms
    if 'Tr' in fileData.keys(): # KITTI visual odometry
        data['T_cam0_velo'] = np.reshape(fileData['Tr'], (3, 4))
    else:
        """
        R0_rect (3x3): Rotation from non-rectified to rectified camera coordinate system
        Tr_velo_to_cam (3x4): Rigid transformation from Velodyne to (non-rectified) camera coordinates
        For not odometry dataset, such as raw/kitti 2015/object, etc
        """
        if 'R0_rect' in fileData.keys():
            R0 = 'R0_rect'
        elif 'R_rect_00' in fileData.keys(): # KITTI raw data
            R0 = 'R_rect_00'
        else:
            raise ValueError

        R_rect_00 = fileData[R0].reshape((3,3))
        Tr_cam0_velo = np.reshape(fileData['Tr_velo_to_cam'], (3,4))
        data['T_cam0_velo'] = R_rect_00 @ Tr_cam0_velo

    T_cam0_velo = np.vstack((data['T_cam0_velo'], [0, 0, 0, 1])) # 4x4

    # project velodyne to cam0 and then translate from cam0 to camN, in 4x4 layout
    data['T_cam0_velo'] = T0.dot(T_cam0_velo)
    data['T_cam1_velo'] = T1.dot(T_cam0_velo)
    data['T_cam2_velo'] = T2.dot(T_cam0_velo)
    data['T_cam3_velo'] = T3.dot(T_cam0_velo)


    """
    Compute the stereo baselines in meters by projecting the origin of each camera frame 
    into the velodyne frame and computing the distances between them
    """
    # the origin point of each camera frame
    p_cam = np.array([0, 0, 0, 1])
    # project each camera frame to the velodyne frame
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    # get baseline of two gray cameras and two rgb cameras
    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0) # baseline of gray cameras
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2) # baseline of rgb cameras

    # get resolution of image, the resolution of all camera is the same
    if 'S_rect_02' in fileData.keys():
        # in (Height, Width) layout
        data['resolution'] = tuple(fileData['S_rect_02'][::-1].astype(np.int32))

    else:
        data['resolution'] = None

    """
    Double check !
    For transformation from a point Y of velodyne to a point Z of rectified camera frame X
    1. translate to camera frame X and then rectify with intrinsic 
        Z = [K_camX|0] @ T_camX_velo @ Y
    2. translate to camera frame 0 and move to camera frame X, then rectify with intrinsic
        Z = P_rect_X0 @ T_cam0_velo @ Y 
    
    i.e. P_rect_X0 = [K_camX|0] @ T_camX_cam0
    """
    for x in range(4):
        T_camX_velo = 'T_cam{}_velo'.format(x)
        K_camX = 'K_cam{}'.format(x)
        P_rect_X0 = 'P_rect_{}0'.format(x)

        Z1 = trans2homo4x4(data[P_rect_X0]) @ data['T_cam0_velo']
        Z2 = trans2homo4x4(data[K_camX]) @ data[T_camX_velo]

        assert np.allclose(Z1, Z2) # Z1, Z2 should be the same

    export_ks = [
        'P_rect_00', 'P_rect_10', 'P_rect_20', 'P_rect_30',
        'T_cam0_velo', 'T_cam1_velo', 'T_cam2_velo', 'T_cam3_velo',
        'K_cam0', 'K_cam1', 'K_cam2', 'K_cam3',
        'b_gray', 'b_rgb',
        'resolution',
    ]

    calibration = dict()
    for k in export_ks:
        calibration[k] = data[k]

    return calibration


