import numpy as np
import matplotlib.pyplot as plt


def disp_map(disp):
    """
    Based on color histogram, convert the gray disp into color disp map.
    The histogram consists of 7 bins, value of each is e.g. [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    Accumulate each bin, named cbins, and scale it to [0,1], e.g. [0.114, 0.299, 0.413, 0.587, 0.701, 0.886, 1.0]
    For each value in disp, we have to find which bin it belongs to
    Therefore, we have to compare it with every value in cbins
    Finally, we have to get the ratio of it accounts for the bin, and then we can interpolate it with the histogram map
    For example, 0.780 belongs to the 5th bin, the ratio is (0.780-0.701)/0.114,
    then we can interpolate it into 3 channel with the 5th [0, 1, 0] and 6th [0, 1, 1] channel-map
    Inputs:
        disp: numpy array, disparity gray map in (Height * Width, 1) layout, value range [0,1]
    Outputs:
        disp: numpy array, disparity color map in (Height * Width, 3) layout, value range [0,1]
    """
    map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])
    # grab the last element of each column and convert into float type, e.g. 114 -> 114.0
    # the final result: [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    bins = map[0:map.shape[0] - 1, map.shape[1] - 1].astype(float)

    # reshape the bins from [7] into [7,1]
    bins = bins.reshape((bins.shape[0], 1))

    # accumulate element in bins, and get [114.0, 299.0, 413.0, 587.0, 701.0, 886.0, 1000.0]
    cbins = np.cumsum(bins)

    # divide the last element in cbins, e.g. 1000.0
    bins = bins / cbins[cbins.shape[0] - 1]

    # divide the last element of cbins, e.g. 1000.0, and reshape it, final shape [6,1]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    cbins = cbins.reshape((cbins.shape[0], 1))

    # transpose disp array, and repeat disp 6 times in axis-0, 1 times in axis-1, final shape=[6, Height*Width]
    ind = np.tile(disp.T, (6, 1))
    tmp = np.tile(cbins, (1, disp.size))

    # get the number of disp's elements bigger than  each value in cbins, and sum up the 6 numbers
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis=0)

    bins = 1 / bins

    # add an element 0 ahead of cbins, [0, cbins]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))
    cbins[1:] = t

    # get the ratio and interpolate it
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s, 0:3] * np.tile(1 - disp, (1, 3)) + map[s + 1, 0:3] * np.tile(disp, (1, 3))

    return disp


def disp_to_color(disp, max_disp=None):
    """
    Transfer disparity map to color map
    Args:
        disp (numpy.array): disparity map in (Height, Width) layout, value range [0, inf]
        max_disp (int): max disparity, optionally specifies the scaling factor
    Returns:
        disparity color map (numpy.array): disparity map in (Height, Width, 3) layout,
            range [0,1]
    """
    # grab the disp shape(Height, Width)
    h, w = disp.shape

    # if max_disp not provided, set as the max value in disp
    if max_disp is None:
        max_disp = np.max(disp)

    # scale the disp to [0,1] by max_disp
    disp = disp.copy() / max_disp

    # reshape the disparity to [Height*Width, 1]
    disp = disp.reshape((h * w, 1))

    # convert to color map, with shape [Height*Width, 3]
    disp = disp_map(disp)

    # convert to RGB-mode
    disp = disp.reshape((h, w, 3))

    return disp



def disp_err_to_color(disp_est, disp_gt):
    """
    Calculate the error map between disparity estimation and disparity ground-truth
    hot color -> big error, cold color -> small error
    Args:
        disp_est (numpy.array): estimated disparity map
            in (Height, Width) layout, range [0,inf]
        disp_gt (numpy.array): ground truth disparity map
            in (Height, Width) layout, range [0,inf]
    Returns:
        disp_err (numpy.array): disparity error map
            in (Height, Width, 3) layout, range [0,1]
    """
    """ matlab
    function D_err = disp_error_image (D_gt,D_est,tau,dilate_radius)
    if nargin==3
      dilate_radius = 1;
    end
    [E,D_val] = disp_error_map (D_gt,D_est);
    E = min(E/tau(1),(E./abs(D_gt))/tau(2));
    cols = error_colormap();
    D_err = zeros([size(D_gt) 3]);
    for i=1:size(cols,1)
      [v,u] = find(D_val > 0 & E >= cols(i,1) & E <= cols(i,2));
      D_err(sub2ind(size(D_err),v,u,1*ones(length(v),1))) = cols(i,3);
      D_err(sub2ind(size(D_err),v,u,2*ones(length(v),1))) = cols(i,4);
      D_err(sub2ind(size(D_err),v,u,3*ones(length(v),1))) = cols(i,5);
    end
    D_err = imdilate(D_err,strel('disk',dilate_radius));
    """
    # error color map with interval (0, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, inf)/3.0
    # different interval corresponds to different 3-channel projection
    cols = np.array(
        [
            [0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
            [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
            [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
            [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
            [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
            [3 / 3.0, 6 / 3.0, 254, 224, 144],
            [6 / 3.0, 12 / 3.0, 253, 174, 97],
            [12 / 3.0, 24 / 3.0, 244, 109, 67],
            [24 / 3.0, 48 / 3.0, 215, 48, 39],
            [48 / 3.0, float("inf"), 165, 0, 38]
        ]
    )

    # [0, 1] -> [0, 255.0]
    disp_est = disp_est.copy() *  255.0
    disp_gt = disp_gt.copy() * 255.0
    # get the error (<3px or <5%) map
    tau = [3.0, 0.05]
    E = np.abs(disp_est - disp_gt)

    not_empty = disp_gt > 0.0
    tmp = np.zeros_like(disp_gt)
    tmp[not_empty] = E[not_empty] / disp_gt[not_empty] / tau[1]
    E = np.minimum(E / tau[0], tmp)

    h, w = disp_gt.shape
    err_im = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    for col in cols:
        y_x = not_empty & (E >= col[0]) & (E <= col[1])
        err_im[y_x] = col[2:]

    # value range [0, 1], shape in [H, W 3]
    err_im = err_im.astype(np.float64) / 255.0

    return err_im

def revalue(map, lower, upper, start, scale):
    mask = (map > lower) & (map <= upper)
    if np.sum(mask) >= 1.0:
        mn, mx = map[mask].min(), map[mask].max()
        map[mask] = ((map[mask] - mn) / (mx -mn + 1e-7)) * scale + start

    return map

def disp_err_to_colorbar(est, gt, with_bar=False, cmap='jet'):
    error_bar_height = 50
    valid = gt > 0
    error_map = np.abs(est - gt) * valid
    h, w= error_map.shape

    maxvalue = error_map.max()
    # meanvalue = error_map.mean()
    # breakpoints = np.array([0, max(1, 1*meanvalue),  max(4, 4*meanvalue),  max(8, min(8*meanvalue, maxvalue/2)) ,  max(12, maxvalue)])
    breakpoints = np.array([0,      1,      2,      4,     12,    16,       max(192, maxvalue)])
    points      = np.array([0,      0.25,   0.38,   0.66,  0.83,  0.95,     1])
    num_bins    = np.array([0,      w//8,   w//8,   w//4,  w//4,  w//8,     w - (w//4 + w//4 + w//8 + w//8 + w//8)])
    acc_num_bins = np.cumsum(num_bins)

    for i in range(1, len(breakpoints)):
        scale = points[i] - points[i-1]
        start = points[i-1]
        lower = breakpoints[i-1]
        upper = breakpoints[i]
        error_map = revalue(error_map, lower, upper, start, scale)

    # [0, 1], [H, W, 3]
    error_map = plt.cm.get_cmap(cmap)(error_map)[:, :, :3]

    if not with_bar:
        return error_map

    error_bar = np.array([])
    for i in range(1, len(num_bins)):
        error_bar = np.concatenate((error_bar, np.linspace(points[i-1], points[i], num_bins[i])))

    error_bar = np.repeat(error_bar, error_bar_height).reshape(w, error_bar_height).transpose(1, 0) # [error_bar_height, w]
    error_bar_map = plt.cm.get_cmap(cmap)(error_bar)[:, :, :3]
    plt.xticks(ticks=acc_num_bins, labels=breakpoints.astype(np.int32))
    # plt.axis('off')

    # [0, 1], [H, W, 3]
    error_map = np.concatenate((error_map, error_bar_map), axis=0)

    return error_map
