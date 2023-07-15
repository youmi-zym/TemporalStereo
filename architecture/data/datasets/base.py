import json
import random
from PIL import Image
import numpy as np
import math

import torch
import torch.nn.functional
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from architecture.modeling.layers import project_to_3d

def pil_loader(filepath):
    """
    open file path as file to avoid ResourceWarning
    https://github.com/python-pillow/Pillow/issues/835
    """
    with open(filepath, 'rb') as fp:
        with Image.open(fp) as img:
            return img.convert('RGB')

def doingNothing(x, *args, **kwargs):
    return x


class AdjustGamma(object):
    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, x):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return transforms.functional.adjust_gamma(x, gamma, gain)


class StereoDatasetBase(Dataset):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.annFile = annFile
        self.root = root
        self.data_list = self.annLoader()

        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.use_common_intrinsics = use_common_intrinsics
        # random give a baseline between stereo rig, please re-init it
        self.baseline = 1.0

        self.do_same_lr_transform = do_same_lr_transform
        self.build_transform()

        self.with_depth_gt = False
        self.with_disp_gt = False
        self.with_flow_gt = False
        self.with_pose_gt = False

    def build_transform(self):
        self.img_loader = pil_loader
        # get an Tensor; if is ByteTensor, it will be normailzed to [0, 1]
        self.to_tensor = transforms.ToTensor()

        # color augmentation
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        self.do_color_aug = self.is_train and random.random() > 0.5
        if self.do_color_aug:
            try:
                self.brightness = (0.4, 2.0)
                self.contrast = (0.5, 1.5)
                self.saturation = (0.5, 1.5)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.Compose([
                    transforms.ColorJitter(
                        self.brightness, self.contrast, self.saturation, self.hue),
                    AdjustGamma(0.8, 1.2, 1.0, 1.0),
                ])
                if not self.do_same_lr_transform:
                    self.color_aug_r = transforms.Compose([
                        transforms.ColorJitter(
                            self.brightness, self.contrast, self.saturation, self.hue),
                        AdjustGamma(0.8, 1.2, 1.0, 1.0),
                    ])

            except:
                print("Cannot Build ColorJitter!!!")
        else:
            self.color_aug = doingNothing
            if not self.do_same_lr_transform:
                self.color_aug_r = doingNothing

    def do_transform(self, sample):
        # PIL object
        H, W = sample[('color', 0, 'l')].height, sample[('color', 0, 'l')].width

        # color augmentation, to tensor, normalize
        for key in list(sample):
            value = sample[key]
            if 'color' in key:
                name, frame_idx, side = key
                # color aug
                if self.do_same_lr_transform:
                    color_aug_value = self.color_aug(value)
                else:
                    if 'l' in key:
                        color_aug_value = self.color_aug(value)
                    elif 'r' in key:
                        if random.random() > 0.5:
                            color_aug_value = self.color_aug_r(value)
                        else:
                            color_aug_value = self.color_aug(value)
                    else:
                        raise ValueError('l or r key is not in sample')

                # value range [0, 1]
                value = self.to_tensor(value)
                color_aug_value = self.to_tensor(color_aug_value)

                # for imagenet normalization, value range [-2.12, 2.64]
                color_aug_value = F.normalize(color_aug_value, mean=self.mean, std=self.std)

                sample[(name, frame_idx, side)] = value
                sample[(name+'_aug', frame_idx, side)] = color_aug_value

        ch = 0; cw = 0; pad_left = 0; pad_right = 0; pad_top = 0; pad_bottom = 0
        if self.height == H and self.width == W:
            return sample
        elif self.is_train:
            assert W >= self.width and H >= self.height, "Image size: ({}, {}) cannot crop to ({}, {})".format(W, H, self.width, self.height)
            ch = random.randint(0, H - self.height)
            cw = random.randint(0, W - self.width)

        elif not self.is_train:
            assert W <= self.width and H <= self.height, "Image size: ({}, {}) cannot pad to ({}, {})".format(W, H, self.width, self.height)
            pad_left = 0
            pad_right = self.width - W
            pad_top = self.height - H
            pad_bottom = 0

        # random crop/pad
        for key in list(sample):
            value = sample[key]
            if self.is_train:
                # random crop
                _include_keys = ['color', 'color_aug', 'disp_gt', 'depth_gt', 'backward_flow_gt', 'forward_flow_gt']
                if key[0] in _include_keys:
                    name, frame_idx, side = key
                    value = value[:, ch:ch+self.height, cw:cw+self.width]

                    # random occlude a region, refer to https://arxiv.org/abs/2106.08486
                    if np.random.binomial(1, 0.5) and 'color_aug' in key and 'r' in key:
                        # [2, 5) -> [2, 4]
                        num = np.random.randint(2, 5)
                        for i in range(num):
                            c = value.shape[0]
                            occw = int(np.random.uniform(50, 250))
                            occh = int(np.random.uniform(50, 180))
                            sw = int(np.random.uniform(0, self.width-occw))
                            sh = int(np.random.uniform(0, self.height-occh))
                            mu, sigma = 0, 0.1
                            noise = np.random.normal(mu, sigma, size=(c, occh, occw))
                            noise = torch.from_numpy(noise)
                            noise = F.normalize(noise, mean=self.mean, std=self.std)

                            mean = value.mean(2).mean(1).view(-1, 1, 1)
                            value[:, sh:sh+occh, sw:sw+occw] = noise + mean-mean

                    sample[(name, frame_idx, side)] = value
            else:
                # pad
                _include_keys = ['color_aug']
                if key[0] in _include_keys:
                    name, frame_idx, side = key
                    # value = pad(value, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=0)
                    value = value.unsqueeze(dim=0)
                    value = torch.nn.functional.interpolate(value, size=(self.height, self.width), mode='bilinear', align_corners=True)
                    value = value.squeeze(dim=0)
                    sample[(name, frame_idx, side)] = value

        return sample

    def annLoader(self):
        data_list = []
        with open(file=self.annFile, mode='r') as fp:
            data_list.extend(json.load(fp))
        return data_list

    def Loader(self, image_path):
        raise NotImplementedError

    def depthLoader(self, depth_path, K=None):
        raise NotImplementedError

    def dispLoader(self, disp_path, K=None):
        raise NotImplementedError

    def flowLoader(self, flow_path):
        raise NotImplementedError

    def extrinsicLoader(self, extrinsic_path):
        raise NotImplementedError

    def getExtrinsic(self, extrinsics, image_path):
        raise NotImplementedError

    def intrinsicLoader(self, intrinsic_path):
        raise NotImplementedError

    def __getitem__(self, idx):
        sample = {}
        item = self.data_list[idx]
        # baseline
        sample['baseline'] = torch.Tensor([self.baseline, ]).float().unsqueeze(dim=0).unsqueeze(dim=0)

        # intrinsics
        # modify it if intrinsic path is different for each image
        if 'intrinsic_path' in item.keys():
            intrinsic_path = item['intrinsic_path']
        else:
            intrinsic_path = item['0']['left_image_path']
        # K is the full_K / (h, w)
        K, full_K, image_resolution = self.intrinsicLoader(intrinsic_path)

        num_scales = min(int(math.log2(self.width)), int(math.log2(self.height)))
        # at the training phase, we use crop rather than resize to get image sample
        # if crop, the intrinsic is the same as original image;
        # if resize, it's linear with the final resolution
        if self.is_train:
            kh, kw = image_resolution
        else:
            kh, kw = self.height, self.width
        for scale in range(num_scales):
            scale_K = np.eye(4)
            scale_K[0, :] = K[0,:] * (kw // (2 ** scale))
            scale_K[1, :] = K[1,:] * (kh // (2 ** scale))
            scale_K[2:, :] = K[2:, :].copy()

            inv_scale_K = np.linalg.pinv(scale_K)

            sample[('K', scale)] = torch.from_numpy(scale_K).float()
            sample[('inv_K', scale)] = torch.from_numpy(inv_scale_K).float()

        # extrinsics
        extrinsics = None
        if self.with_pose_gt and 'extrinsic_path' in item:
            extrinsics = self.extrinsicLoader(item['extrinsic_path'])

        for i, frame_idx in enumerate(sorted(self.frame_idxs)):
            curitem = item[str(frame_idx)]
            sample[('color', frame_idx, 'l')] = self.Loader(curitem['left_image_path'])
            sample[('color', frame_idx, 'r')] = self.Loader(curitem['right_image_path'])
            if extrinsics is not None:
                # the transformation matrix is from world original point to current frame
                left_T, left_inv_T, right_T, right_inv_T = self.getExtrinsic(extrinsics, curitem['left_image_path'])
                sample[('T', frame_idx, 'l')] = left_T.float()
                sample[('inv_T', frame_idx, 'l')] = left_inv_T.float()
                sample[('T', frame_idx, 'r')] = right_T.float()
                sample[('inv_T', frame_idx, 'r')] = right_inv_T.float()

            if 'left_depth_path' in curitem.keys() and self.with_depth_gt:
                depth_gt, disp_gt = self.depthLoader(curitem['left_depth_path'], full_K)
                sample[('depth_gt', frame_idx, 'l')] = depth_gt
                sample[('disp_gt', frame_idx, 'l')] = disp_gt
            if 'right_depth_path' in curitem.keys() and self.with_depth_gt:
                depth_gt, disp_gt = self.depthLoader(curitem['right_depth_path'], full_K)
                sample[('depth_gt', frame_idx, 'r')]  = depth_gt
                sample[('disp_gt', frame_idx, 'r')] = disp_gt

            if 'left_disp_path' in curitem.keys() and self.with_disp_gt:
                depth_gt, disp_gt = self.dispLoader(curitem['left_disp_path'], full_K)
                if ('depth_gt', frame_idx, 'l') not in sample:
                    sample[('depth_gt', frame_idx, 'l')] = depth_gt
                sample[('disp_gt', frame_idx, 'l')] = disp_gt
            if 'right_disp_path' in curitem.keys() and self.with_disp_gt:
                depth_gt, disp_gt = self.dispLoader(curitem['right_disp_path'], full_K)
                if ('depth_gt', frame_idx, 'r') not in sample:
                    sample[('depth_gt', frame_idx, 'r')] = depth_gt
                sample[('disp_gt', frame_idx, 'r')] = disp_gt

            # no previous frame
            if i > 0 and 'left_backward_flow_path' in curitem.keys() and self.with_flow_gt:
                sample[('backward_flow_gt', frame_idx, 'l')] = self.flowLoader(curitem['left_backward_flow_path'])
            if i > 0 and 'right_backward_flow_path' in curitem.keys() and self.with_flow_gt:
                sample[('backward_flow_gt', frame_idx, 'r')] = self.flowLoader(curitem['right_backward_flow_path'])
            # no next frame
            if i < len(self.frame_idxs)-1 and 'left_forward_flow_path' in curitem.keys() and self.with_flow_gt:
                sample[('forward_flow_gt', frame_idx, 'l')] = self.flowLoader(curitem['left_forward_flow_path'])
            if i < len(self.frame_idxs)-1 and 'right_forward_flow_path' in curitem.keys() and self.with_flow_gt:
                sample[('forward_flow_gt', frame_idx, 'r')] = self.flowLoader(curitem['right_forward_flow_path'])

        sample = self.do_transform(sample)

        return sample

    def __len__(self):
        return len(self.data_list)

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Is train: {}\n'.format(self.is_train)
        repr_str += ' ' * 4 + 'Root: {}\n'.format(self.root)
        repr_str += ' ' * 4 + 'annFile: {}\n'.format(self.annFile)
        repr_str += ' ' * 4 + 'Length: {}\n'.format(self.__len__())
        repr_str += ' ' * 4 + 'Image with resolution: ({}, {})\n'.format(self.height, self.width)
        repr_str += ' ' * 4 + 'Frame indexes: {}\n'.format(self.frame_idxs)
        repr_str += ' ' * 4 + 'Use common intrinsic: {}\n'.format(self.use_common_intrinsics)
        repr_str += ' ' * 4 + 'normalize with mean: {}, std: {}\n'.format(self.mean, self.std)
        repr_str += ' ' * 4 + 'Load depth map: {}\n'.format(self.with_depth_gt)
        repr_str += ' ' * 4 + 'Load disparity map: {}\n'.format(self.with_disp_gt)
        repr_str += ' ' * 4 + 'Load flow map: {}\n'.format(self.with_flow_gt)

        return repr_str

    @property
    def name(self):
        raise NotImplementedError
