import sys
import os

sys.path.insert(0, '/home/yzhang/projects/mono/StereoBenchmark/')
from architecture.data.datasets import VKITTI2StereoDataset
from architecture.data.datasets import SceneFlowStereoDataset
from architecture.data.datasets import TARTANAIRStereoDataset
from architecture.data.datasets import KITTI2015StereoDataset
from architecture.data.datasets import KITTIRAWStereoDataset

def build_stereo_dataset(cfg, phase):

    data_root = cfg.DATA_ROOT
    data_type = cfg.TYPE
    annFile = cfg.ANNFILE
    height = cfg.HEIGHT
    width = cfg.WIDTH
    frame_idxs = cfg.FRAME_IDXS
    use_common_intrinsics = cfg.get('USE_COMMON_INTRINSICS', True)
    do_same_lr_transform = cfg.get('DO_SAME_LR_TRANSFORM', True)
    mean = cfg.get('MEAN', (0.485, 0.456, 0.406))
    std = cfg.get('STD', (0.229, 0.224, 0.225))


    is_train = True if phase == 'train' else False

    if 'VKITTI2' in data_type:
        dataset = VKITTI2StereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                       do_same_lr_transform, mean, std)

    elif 'SceneFlow' in data_type:
        dataset = SceneFlowStereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)

    elif 'TartanAir' in data_type:
        dataset = TARTANAIRStereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)

    elif 'KITTI2015' in data_type:
        dataset = KITTI2015StereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)

    elif 'KITTIRAW' in data_type:
        dataset = KITTIRAWStereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)

    else:
        raise ValueError("invalid data type: {}".format(data_type))

    return dataset


if __name__ == '__main__':
    """
    Test the Stereo Dataset
    """
    import sys
    sys.path.insert(0, '/home/yzhang/projects/stereo/TemporalStereo/')
    import matplotlib.pyplot as plt

    from projects.TemporalStereo.config import get_cfg, get_parser
    args = get_parser().parse_args()
    args.config_file = '/home/yzhang/projects/stereo/TemporalStereo/projects/TemporalStereo/configs/kitti2015.yaml'
    cfg = get_cfg(args)

    dataset = build_stereo_dataset(cfg.DATA.TRAIN, 'train')

    print(dataset)

    print("Dataset contains {} items".format(len(dataset)))

    idxs = [0, ] # 100, 1000]
    for i in idxs:
        sample = dataset[i]
        for key in list(sample):
            _include_keys = ['color_aug', 'color', 'disp_gt', 'depth_gt']
            if key[0] in _include_keys:
                print('Key {} with shape: {}'.format(key, sample[key].shape))
                img = sample[key].permute(1, 2, 0).cpu().numpy()
                plt.imshow(img)
                plt.show()

    print('Done!')


