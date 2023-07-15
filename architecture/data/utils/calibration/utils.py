import numpy as np


# Reference include:
#   https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py

def cart_to_homo(pts_3d):
    """
    Convert Cartesian to Homogeneous
    Inputs:
        pts_3d, (numpy.ndarray): 3xn points in Cartesian
    Outputs:
        pts_4d, (numpy.ndarray): 4xn points in Homogeneous by pending 1
    """
    c,n = pts_3d.shape
    assert c == 3
    pts_4d = np.vstack((pts_3d, np.ones(n)))
    assert pts_4d.shape == (4, n)
    return pts_4d

def trans2homo4x4(T):
    """
    Transform into homogeneous matrix
    Inputs:
        T, (numpy.ndarray): 3x3 or 3x4 matrix
    Outputs:
        homoT, (numpy.ndarray): 4x4 matrix
    """
    h, w = T.shape
    assert h<=4 and w <=4
    homoT = np.eye(4)
    homoT[:h, :w] = T
    return homoT

def load_velodyne(filepath,
                  no_reflect=False,
                  load_bin_without_reflect=False,
                  dtype=np.float32):
    """
    velodyne point cloud contains 4 values, where the first 3 values corresponds to x, y, z,
    and the last value is the reflectance information, often it's not used
    Args:
        filepath, (str): the velodyne file path
        no_reflect, (bool): weather return with the 4th value, i.e., reflectance information
        load_bin_without_reflect, (bool): weather load the 4th value, i.e., reflectance information
        dtype, (str or dtype): typecode or data-type to which the array is cast.

    Outputs:
        velo, (dtype): the loaded velodyne point cloud, in nx3 or nx4 layout

    """
    channels = 3 if load_bin_without_reflect else 4
    velo = np.fromfile(filepath, dtype=dtype).reshape(-1, channels)
    if no_reflect:
        return velo[:, :3]
    else:
        if channels == 3:
            velo = cart_to_homo(velo.transpose()).transpose() # fake reflectance
        return velo