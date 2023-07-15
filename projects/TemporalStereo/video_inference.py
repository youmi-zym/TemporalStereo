import sys
sys.path.insert(0, '/home/yzhang/projects/stereo/TemporalStereo/')
import matplotlib
matplotlib.use('TkAgg')
import cv2
import skimage
import skimage.io
import skimage.transform

import matplotlib.pyplot as plt
large = 22; med = 16; small = 12
params = {
    'axes.titlesize': large,
    'legend.fontsize': med,
    'figure.figsize': (12, 20),
    'axes.labelsize': med,
    'xtick.labelsize': med,
    'ytick.labelsize': med,
    'figure.titlesize': small,
}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')

import matplotlib.axes
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from collections import abc as container_abcs
from tqdm import tqdm
import pickle

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as transF
from architecture.utils.config import CfgNode
from architecture.utils.visualization import colormap, disp_err_to_color, disp_err_to_colorbar, disp_to_color
from architecture.data.utils.load_tartanair import read_tartanair_extrinsic

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(filepath):
    """
    open file path as file to avoid ResourceWarning
    https://github.com/python-pillow/Pillow/issues/835
    """
    with open(filepath, 'rb') as fp:
        with Image.open(fp) as img:
            img = img.convert('RGB')

            return img


def read_extrinsic(extrinsic_fn, inverse=True, use_gt=False):
    """
    We assume the extrinsic is obtained by ORBSLAM3,
    The pose of it is from camera to world, but we need world to camera, so we have to inverse it
    """
    if use_gt:
        data = read_tartanair_extrinsic(extrinsic_fn, side='left')
    else:
        lineid = 0
        data = {}
        with open(extrinsic_fn, 'r') as fp:
            for line in fp.readlines():
                values = line.rstrip().split(' ')
                frame = '{:02d}'.format(lineid)
                camera = '02'
                key = 'T_cam{}'.format(camera)
                inv_key = 'inv_T_cam{}'.format(camera)
                matrix = np.array([float(values[i]) for i in range(len(values))])
                matrix = matrix.reshape(3, 4)
                matrix = np.concatenate((matrix, np.array([[0, 0, 0, 1]])), axis=0)
                if inverse:
                    item = {
                        # inverse pose
                        key: np.linalg.pinv(matrix),
                        inv_key: matrix,
                    }
                else:
                    item = {
                        inv_key: np.linalg.pinv(matrix),
                        key: matrix,

                    }
                data['Frame{}:{}'.format(frame, camera)] = item
                lineid += 1

    return data

def read_image(image_fn, resize_to_shape=None):
    image = pil_loader(image_fn)
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    # for imagenet normalization, value range [-2.12, 2.64]
    image_proc = transF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if resize_to_shape is not None:
        image_proc = F.interpolate(image_proc.unsqueeze(dim=1), size=resize_to_shape, mode='bilinear', align_corners=True).squeeze(dim=1)

    return image, image_proc

def read_intrinsics(intrinsic_fn=None, device_type='zed2'):
    if device_type == 'zed2':
        # zed2 intrinsics
        full_h, full_w = 1080, 1920
        K = np.array([[1116.0751953125 / full_w, 0, 949.7744140625 / full_w, 0],
                      [0, 1116.0751953125 / full_h, 533.137939453125 / full_h, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        baseline = 0.11974537372589111
    elif device_type == 'kitti':
        full_h, full_w = 375, 1242
        K=np.array([[721.5377/full_w,      0,                  609.5593/full_w,    0],
                    [0,                    721.5377/full_h,    172.854/full_h,     0],
                    [0,                    0,                  1,                  0],
                    [0,                    0,                  0,                  1]])
        baseline = 0.54

    elif device_type == 'tartanair':
        full_h, full_w = 480, 640
        K=np.array([[320/full_w,      0,            320/full_w,    0],
                    [0,               320/full_h,   240/full_h,    0],
                    [0,               0,            1,             0],
                    [0,               0,            0,             1]])
        baseline = 0.25

    else:
        K = np.eye(4)
        baseline = 1.0

    return K, baseline

def read_disparity(disp_fn):
    disp = cv2.imread(disp_fn, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    valid_mask = disp > 0
    disp = (disp*valid_mask) / 256.0

    return disp


def to_cpu(tensor):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_cpu(tensor[key]) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_cpu(samples) for samples in tensor]
    else:
        return tensor

    # raise TypeError((error_msg.format(type(tensor))))

def to_gpu(tensor, device):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device).unsqueeze(0)
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_gpu(tensor[key], device) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_gpu(samples, device) for samples in tensor]
    else:
        return tensor

    # raise TypeError((error_msg.format(type(tensor))))

def visualize(outputs, image_name, log_dir):
    left_image = outputs[('color', 0, 'l')][0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    right_image = outputs[('color', 0, 'r')][0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    h, w, c = left_image.shape
    disp_gt = outputs.get(('disp_gt', 0, 'l'), None)
    if disp_gt is not None:
        disp_gt = disp_gt[0, 0].cpu().numpy()


    # removing pad in estimation
    gh, gw = h, w
    disp_est = outputs[('disps', 0, 'l')][0]
    ph, pw = disp_est.shape[-2:]
    disp_est = F.interpolate(disp_est * gw / pw, size=(gh, gw), mode='bilinear', align_corners=True)
    disp_est = disp_est[0, 0,].cpu().numpy()

    # calc error
    perct, epe = 0, 0
    if disp_gt is not None:
        valid = (disp_gt > 0) & (disp_gt < 192)
        valid = valid.astype(np.float64)
        total_num = valid.sum()
        err = (np.abs(disp_gt - disp_est) * valid.astype(np.float64)) > 3
        perct = err.astype(np.float64).sum() / total_num
        perct = perct * 100

        err = (np.abs(disp_gt - disp_est) * valid.astype(np.float64))
        epe = err.astype(np.float64).sum() / total_num

        print('3PE: {:.2f}% \n EPE: {:.3f}'.format(perct, epe))

    if disp_gt is not None:
        cat_disp = np.concatenate((disp_est, disp_gt), axis=0)
        cat_disp_color = disp_to_color(cat_disp).clip(0, 1)
        error_map = colormap(disp_err_to_color, disp_est, disp_gt, normalize=False).clip(0, 1)
        error_map_with_bar = colormap(disp_err_to_colorbar, disp_est, disp_gt, normalize=False, with_bar=True, cmap='jet').clip(0, 1)
        cats = np.concatenate((left_image, right_image, cat_disp_color, error_map, error_map_with_bar), axis=0)
    else:
        cat_disp = disp_est.copy()
        cat_disp_color = disp_to_color(cat_disp).clip(0, 1)
        cats = np.concatenate((left_image, right_image, cat_disp_color), axis=0)

    cats_dir = os.path.join(log_dir, 'cats')
    os.makedirs(cats_dir, exist_ok=True)
    plt.imshow(cats)
    plt.title('3PE: {:.2f}%, EPE: {:.3f}'.format(perct, epe))
    plt.savefig(os.path.join(cats_dir, image_name.split('.')[0]+'.png'))
    plt.close()

    submit_dir = os.path.join(log_dir, 'disp_0')
    os.makedirs(submit_dir, exist_ok=True)
    skimage.io.imsave(os.path.join(submit_dir, image_name.split('.')[0]+'.png'), (disp_est * 256).astype('uint16'))

    color_dir = os.path.join(log_dir, 'color_disp')
    os.makedirs(color_dir, exist_ok=True)
    color_disp = disp_to_color(disp_est).clip(0, 1)
    plt.imsave(os.path.join(color_dir, image_name.split('.')[0]+'.png'), color_disp, cmap='hot')

    return perct, epe

def inference_stereo(
        model: nn.Module,
        data_root: str,
        resize_to_shape: tuple,
        log_dir: str,
        device: str = 'cpu',
        perform_visualization: bool = True,
):
    device = torch.device(device)
    model = model.to(device)

    imgLists = os.listdir(os.path.join(data_root, 'left'))
    imgLists = [img for img in imgLists if is_image_file(img)]
    imgLists.sort()
    # extrinsic_path = os.path.join(data_root, 'orbslam3_pose.txt')
    # extrinsics = read_extrinsic(extrinsic_path, inverse=False, use_gt=False)
    extrinsic_path = os.path.join(data_root, 'pose_left.txt')
    extrinsics = read_extrinsic(extrinsic_path, inverse=False, use_gt=True)
    norm_K, baseline = read_intrinsics(device_type='tartanair')
    kh, kw = resize_to_shape
    scale_K = np.eye(4)
    scale_K[0, :] = norm_K[0, :] * kw
    scale_K[1, :] = norm_K[1, :] * kh
    scale_K[2:, :] = norm_K[2:, :].copy()
    inv_scale_K = np.linalg.pinv(scale_K)

    last_outputs = {
        ('prev_info', -1, 'l'): {
            'prev_disp': to_gpu(torch.zeros((1, resize_to_shape[0], resize_to_shape[1])), device=device),
        }
    }
    img_id = 0
    percts = {}
    epes = {}
    for img_name in tqdm(imgLists):
        left_image, left_image_proc = read_image(os.path.join(data_root, 'left', img_name), resize_to_shape)
        right_image, right_image_proc = read_image(os.path.join(data_root, 'right', img_name), resize_to_shape)

        left_T = torch.from_numpy(extrinsics['Frame{:02d}:02'.format(img_id)]['T_cam02'])
        left_inv_T = torch.from_numpy(extrinsics['Frame{:02d}:02'.format(img_id)]['inv_T_cam02'])
        if img_id == 0:
            last_left_T = left_T
            last_left_inv_T = left_inv_T
        else:
            last_left_T = torch.from_numpy(extrinsics['Frame{:02d}:02'.format(img_id-1)]['T_cam02'])
            last_left_inv_T = torch.from_numpy(extrinsics['Frame{:02d}:02'.format(img_id-1)]['inv_T_cam02'])

        # left_T = torch.eye(4)
        # left_inv_T = torch.eye(4)
        # last_left_T = torch.eye(4)
        # last_left_inv_T = torch.eye(4)

        batch = {
            ('color', 0, 'l'): left_image,
            ('color', 0, 'r'): right_image,
            ('color_aug', 0, 'l'): left_image_proc,
            ('color_aug', 0, 'r'): right_image_proc,
            ('T', 0, 'l'): left_T.float(),
            ('inv_T', 0, 'l'): left_inv_T.float(),
            ('T', -1, 'l'): last_left_T.float(),
            ('inv_T', -1, 'l'): last_left_inv_T.float(),
            ('K', 0): torch.from_numpy(scale_K).float(),
            ('inv_K', 0): torch.from_numpy(inv_scale_K).float(),
            ('baseline'): torch.Tensor([baseline, ]).float().unsqueeze(dim=0).unsqueeze(dim=0)
        }

        disp_gt_path = os.path.join(data_root, 'disp_gt', img_name)
        if os.path.exists(disp_gt_path):
            disp_gt = read_disparity(disp_gt_path)
            batch[('disp_gt', 0, 'l')] = torch.from_numpy(disp_gt.astype(np.float32)).unsqueeze(0)

        batch = to_gpu(batch, device)
        print("Start processing: {}...".format(img_name))

        with torch.no_grad():
            outputs = model(batch, last_outputs, is_train=False, timestamp=0)

        outputs.update(batch)
        last_outputs = {}
        last_outputs[('prev_info', -1, 'l')] = outputs[('prev_info', 0, 'l')]

        outputs = to_cpu(outputs)

        if perform_visualization:
            perct, epe = visualize(outputs, img_name, log_dir)
            percts[img_id] = perct
            epes[img_id] = epe

        img_id += 1

    print("Writing {} errors to: {}".format(img_id, os.path.join(log_dir, 'error.txt')))
    total_epe, total_perct = 0.0, 0.0
    with open(os.path.join(log_dir, 'error.txt'), 'w') as fp:
        for i in range(img_id):
            fp.write("{:04d}: {:.4f} {:.4f}\n".format(i, epes[i], percts[i]))
            total_epe += epes[i]
            total_perct += percts[i]

    print('Sequence average EPE: {:.4f}, 3PE: {:.4f}'.format(total_epe/img_id, total_perct/img_id))

    with open(os.path.join(log_dir, 'error.txt'), 'a') as fp:
        fp.write('Sequence average EPE: {:.4f}, 3PE: {:.4f}'.format(total_epe/img_id, total_perct/img_id))


    print("Inference Done!")


if __name__ == '__main__':
    print("Start Inference Video ... ")

    parser = argparse.ArgumentParser("Video Inference")

    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="path to checkpoint",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        help="data root",
        default='./demo_data/',
    )

    parser.add_argument(
        "--device",
        type=str,
        help="device for running, e.g., cpu, cuda:0",
        default="cuda:0"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        help="directory path for logging",
        default='./output/'
    )

    parser.add_argument(
        "--resize-to-shape",
        nargs="+",
        type=int,
        help="image shape after padding for inference, e.g., [544, 960],"
             "after inference, result will crop to original image size",
        default=[384, 1280],
    )

    from projects.TemporalStereo.TemporalStereo import TemporalStereo
    from projects.TemporalStereo.config import get_cfg
    args = parser.parse_args()
    cfg = get_cfg(args)
    model = TemporalStereo(cfg.convert_to_dict())

    data_root = args.data_root

    checkpoint_path = args.checkpoint_path
    assert os.path.isfile(checkpoint_path)
    print('Load checkpoint from: ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    resize_to_shape = args.resize_to_shape
    print("Image will pad to shape: ", resize_to_shape)

    device = args.device
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    print("Result will save to ", log_dir)

    print("Start Inference ... ")
    inference_stereo(
        model,
        data_root,
        resize_to_shape,
        log_dir,
        device,
        perform_visualization=True,
    )

    print("Done!")
