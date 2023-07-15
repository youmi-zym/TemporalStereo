import numpy as np

# Reference include:
#   https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py


class Projection(object):
    '''
    Calibration matrices and utils
    3d XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.

    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
    x_ref = Tr_velo_to_cam * x_velo
    x_rect = R0_rect * x_ref
    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]
            = K * [1|t]
    image2 coord:
    ----> x-axis (u)
    |
    |
    v y-axis (v)
    velodyne coord:
    front x, left y, up z
    rect/ref camera coord:
    right x, down y, front z
    Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

    Args:
        calib, (dict): the loaded calibration information
        resolution, (tuple): the image resolution, in (H,W) layout
        cam_id, (str, int): the id of camera frame
    '''
    def __init__(self, calib, resolution=(1242, 375), cam_id='2'):
        from .utils import cart_to_homo, trans2homo4x4

        # Camera intrinsic matrix, in 3x3 layout
        self.K_camX = calib['K_cam{}'.format(cam_id)]
        # Rigid transform from Velodyne coord to camera coord, in 4x4 layout
        self.T_camX_velo = calib['T_cam{}_velo'.format(cam_id)]
        # Image resolution
        if calib['resolution'] is not None and resolution is not None:
            ch, cw = calib['resolution']
            gh, gw = resolution
            if (ch != gh) or (cw !=gw):
                print(
                    'WARNING, given resolution ({},{}) is not the same as resoltion ({}, {}) from calibration file,\n'
                    'Resolution will set as the data from calibration file: ({}, {})!'.format(gh, gw, ch, cw, ch, cw)
                )
            self.h, self.w = (ch, cw)
        elif resolution is not None:
            self.h, self.w = resolution
        elif calib['resolution'] is not None:
            self.h, self.w = calib['resolution']
        else:
            raise ValueError

        if 'b_rgb' not in calib: # in meter
            print(
                'WARNING, there is no baseline given, baseline=0.54 for rgb cameras is setting here!'
            )
            self.b_rgb = 0.54
        else:
            self.b_rgb = calib['b_rgb']

        # Rigid transform from camera coord to Velodyne coord, in 4x4 layout
        self.T_velo_camX = np.linalg.inv(self.T_camX_velo)
        # Projection matrix from rect camera coord to image coord, in 4x4 layout
        self.T_image_camX = trans2homo4x4(self.K_camX)
        # Projection matrix from image coord to rect camera coord, in 4x4 layout
        self.T_camX_image = np.linalg.inv(self.T_image_camX)

    def update_resolution(self, resolution):
        """
        Update resolution
        Inputs:
            resolution, (tuple): the image resolution, in (H,W) layout
        """
        self.h, self.w = resolution

    @classmethod
    def gen_projection(cls,
                       K_cam,
                       T_cam_velo=np.eye(4),
                       resolution=(1242, 375),
                       b_rgb=None):
        calib = dict()
        cam_id = '2'
        calib['K_cam{}'.format(cam_id)] = K_cam
        calib['T_cam{}_velo'.format(cam_id)] = T_cam_velo
        if b_rgb is not None:
            calib['b_rgb'] = b_rgb
        return cls(calib, resolution, cam_id)

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_rect(self, pts_3d_velo):
        """
        Inputs:
            pts_3d_velo, (numpy.ndarray): in velodyne coord, in (3, N) layout, in Cartesian
        Outputs:
            pts_3d_rect, (numpy.ndarray): in camera coord, in (3, N) layout, in Cartesian
        """
        from architecture.data.utils.calibration import cart_to_homo

        pts_3d_rect = self.T_camX_velo @ cart_to_homo(pts_3d_velo)
        return pts_3d_rect[:3, :]

    def project_rect_to_velo(self, pts_3d_rect):
        """
        Inputs:
            pts_3d_rect, (numpy.ndarray): in camera coord, in (3, N) layout, in Cartesian
        Outputs:
            pts_3d_velo, (numpy.ndarray): in velodyne coord, in (3, N) layout, in Cartesian
        """
        from .utils import cart_to_homo

        pts_3d_velo = self.T_velo_camX @ cart_to_homo(pts_3d_rect)
        return pts_3d_velo[:3, :]

    # ===========================
    # ------- 3d to 2d image ----
    # ===========================
    def project_rect_to_depth(self, pts_3d_rect):
        """
        Project rectified 3d points to generate a depth map
        Inputs:
            pts_3d_rect, (numpy.ndarray): in camera coord, in (3, N) layout, in Cartesian
        Outputs:
            depth_map, (numpy.ndarray): in image coord, the projected depth map, in (H, W) layout

        """
        assert pts_3d_rect.shape[0] == 3
        y, x , depth, sel = self.project_rect_to_image(pts_3d_rect)
        depth_map = np.zeros((self.h, self.w), dtype=np.float32)
        depth_map[y[sel], x[sel]] = depth[sel]
        return depth_map

    def project_rect_to_image(self, pts_3d_rect):
        """
        Project rectified 3d points to image plane.
        Inputs:
            pts_3d_rect, (numpy.ndarray): in camera coord, in (3, N) layout, in Cartesian
        Outputs:
            y, (numpy.ndarray): the index for y-axis, in (N,) layout, dtype = int64
            x, (numpy.ndarray): the index for x-axis, in (N,) layout, dtype = int64
            depth, (numpy.ndarray): the depth values, in (N,) layout, dtype = float64
            sel, (numpy.ndarray): the selected index, in (N,) layout, dtype = bool

        """
        assert pts_3d_rect.shape[0] == 3

        x_y_depth = self.K_camX @ pts_3d_rect
        # (x,y,depth) -> (x/depth, y/depth, 1)
        depth = x_y_depth[2, :]
        x_y_depth[:2, :] /= depth

        x = np.rint(x_y_depth[0, :]).astype(np.int)
        y = np.rint(x_y_depth[1, :]).astype(np.int)

        # get valid points
        sel = ((x>=0) & (x<self.w) & (y>=0) & (y<self.h) & (depth>0))

        # find the duplicate points and choose the closet depth
        valid_set = dict()
        for i, (xx, yy) in enumerate(zip(x, y)):
            if sel[i] == False:
                continue
            if valid_set.get((xx, yy)) is None:
                valid_set[(xx, yy)] = i
            else:
                idx = valid_set[(xx, yy)]
                if depth[i] < depth[idx]: # current point has closer depth
                    sel[idx] = False
                    valid_set[(xx, yy)] = idx
                else:
                    sel[i] = False

        return y, x, depth, sel

    def project_velo_to_depth(self, pts_3d_velo):
        """
        Project original 3d points to generate a depth map
        Inputs:
            pts_3d_velo, (numpy.ndarray): in velodyne coord, in (3, N) layout, in Cartesian
        Outputs:
            depth_map, (numpy.ndarray): in image coord, the projected depth map, in (H, W) layout

        """
        assert pts_3d_velo.shape[0] == 3
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        depth_map = self.project_rect_to_depth(pts_3d_rect)
        return depth_map

    # ===========================
    # ------- 2d to 2d ----------
    # ===========================
    def depth_to_disp(self, depth_map):
        """
        Covert depth map to disparity map
        Inputs:
            depth_map, (numpy.ndarray): in image coord, in (H, W) layout
        Outputs:
            disp_map, (numpy.ndarray): in image coord, in (H, W) layout
        """
        disp_map = np.zeros_like(depth_map)
        y, x = np.where(depth_map > 0.0)
        disp_map[y, x] = self.K_camX[0, 0] * self.b_rgb / depth_map[y, x] # f_x * baseline / depth
        return disp_map

    def disp_to_depth(self, disp_map):
        """
        Covert disparity map to depth map
        Inputs:
            disp_map, (numpy.ndarray): in image coord, in (H, W) layout
        Outputs:
            depth_map, (numpy.ndarray): in image coord, in (H, W) layout

        """
        depth_map = np.zeros_like(disp_map)
        y, x = np.where(disp_map > 0.0)
        depth_map[y, x] = self.K_camX[0, 0] * self.b_rgb / disp_map[y, x] # f_x * baseline / disp
        return depth_map

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def depth_to_rect_with_color(self, depth_map, image=None):
        """
        reverse project depth map to rectified image with color
        Inputs:
            depth (numpy.ndarray): in image coord, in (H, W) layout
            image (optional, numpy.ndarray): in (H, W, C) layout
        Outputs:
            pts_3d_rect (numpy.ndarray): in camera coord, in (3, N) layout, in Cartesian
            color (numpy.ndarray): the color of corresponding depth point, in (N, C) layout
        """
        assert len(depth_map.shape) == 2 # (H, W)
        if image is not None and (self.h, self.w) != tuple(image.shape[:2]):
            print(
                'WARNING, the shape of depth map ({}, {}) is not coincide with image shape({}, {})!'.format(
                    depth_map.shape[0], depth_map.shape[1], image.shape[0], image.shape[1]
                )
            )
        y, x = np.where(depth_map > 0.0)
        x_y_1 = np.vstack((x, y, np.ones_like(x, dtype=np.int))) # (3,N)
        x_y_depth = x_y_1 * depth_map[y, x]
        pts_3d_rect = np.linalg.inv(self.K_camX) @ x_y_depth # (3, N)

        if image is not None:
            color = image[y, x] # (N, C)
        else:
            color = None

        return pts_3d_rect, color
