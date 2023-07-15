import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from collections import abc as container_abcs
from tqdm import tqdm
import pickle

from architecture.data.datasets import build_stereo_dataset
from architecture.utils.config import CfgNode
from architecture.utils.visualization import colormap, disp_err_to_color, disp_err_to_colorbar, disp_to_color

from TemporalStereo import TemporalStereo
from config import get_cfg


def to_cpu(tensor):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_cpu(tensor[key]) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_cpu(samples) for samples in tensor]
    elif tensor is None:
        return tensor

    raise TypeError((error_msg.format(type(tensor))))

def to_gpu(tensor, device):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device).unsqueeze(0)
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_gpu(tensor[key], device) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_gpu(samples, device) for samples in tensor]

    raise TypeError((error_msg.format(type(tensor))))

def visualize(outputs, image_name, log_dir):
    left_image = outputs[('color', 0, 'l')][0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    right_image = outputs[('color', 0, 'r')][0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    h, w, c = left_image.shape
    image_resolution = (h, w)
    disp_gt = outputs.get(('disp_gt', 0, 'l'), None)
    if disp_gt is not None:
        disp_gt = disp_gt[0, 0].cpu().numpy()

    # removing pad in estimation
    disp_est = outputs[('disps', 0, 'l')][0]
    gh, gw = image_resolution
    ph, pw = disp_est.shape[-2:]
    disp_est = F.interpolate(disp_est * gw / pw, size=(gh, gw), mode='bilinear', align_corners=True)
    disp_est = disp_est[0, 0,].cpu().numpy()

    # calc error
    perct = 0
    epe = 0
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
    plt.title('3PE: {:.2f}% \n EPE: {:.3f}'.format(perct, epe))
    plt.savefig(os.path.join(cats_dir, image_name.split('.')[0]+'.png'))
    plt.close()

    submit_dir = os.path.join(log_dir, 'disp_0')
    os.makedirs(submit_dir, exist_ok=True)
    skimage.io.imsave(os.path.join(submit_dir, image_name.split('.')[0]+'.png'), (disp_est * 256).astype('uint16'))

    color_dir = os.path.join(log_dir, 'color_disp')
    os.makedirs(color_dir, exist_ok=True)
    color_disp = disp_to_color(disp_est).clip(0, 1)
    plt.imsave(os.path.join(color_dir, image_name.split('.')[0]+'.png'), color_disp, cmap='hot')

    return perct


def inference_stereo(
        model: nn.Module,
        dataset,
        log_dir: str,
        device: str = 'cpu',
        perform_visualization: bool = True,
):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    error = 0
    cnt = 0

    for batch_idx, batch in tqdm(enumerate(dataset)):
        image_name = dataset.data_list[batch_idx]['0']['left_image_path'].split('/')[-1]
        print('Processing Image ID: {}'.format(image_name))
        batch = to_gpu(batch, device)

        assert not  model.training
        with torch.no_grad():
            outputs = model.multi_frame_forward(batch, is_train=False)
        assert not  model.training

        outputs.update(batch)

        outputs = to_cpu(outputs)

        if perform_visualization:
            error += visualize(outputs, image_name, log_dir)
            cnt += 1

    print('Average 3PE: {:.3f}%'.format(error / cnt)) # 0.596% noeval: 1.033% # 0.664%  # 1.181%

    print("Inference Done!")


if __name__ == '__main__':
    print("Start Inference TemporalStereo for Submission ... ")

    parser = argparse.ArgumentParser("TemporalStereo Submission")

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
        "--annfile",
        type=str,
        help="path to annotation file",
        default='',
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
        help="image shape after resize for inference, e.g., [544, 960],"
             "after inference, result will resize to original image size",
        default=[384, 1280],
    )

    args = parser.parse_args()
    cfg = get_cfg(args)
    model = TemporalStereo(cfg.convert_to_dict())

    checkpoint_path = args.checkpoint_path
    os.path.isfile(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    resize_to_shape = args.resize_to_shape
    print("Image will resize to shape: ", resize_to_shape)

    print("Start Preparing Data ... ")
    data_root = args.data_root
    annfile = args.annfile
    data_config = CfgNode()
    data_config.DATA_ROOT = data_root
    data_config.TYPE = "KITTI2015"
    data_config.ANNFILE = annfile
    data_config.HEIGHT = resize_to_shape[0]
    data_config.WIDTH = resize_to_shape[1]
    data_config.FRAME_IDXS = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0,]

    dataset = build_stereo_dataset(data_config, phase='val')

    device = args.device
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    print("Result will save to ", log_dir)

    print("Start Inference ... ")
    inference_stereo(
        model,
        dataset,
        log_dir,
        device,
        perform_visualization=True,
    )

    print("Done!")