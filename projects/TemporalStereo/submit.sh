#!/bin/bash
root=/home/faleotti/projects/exps/TemporalStereo/kitti2015v2/temporal

ann=./splits/view_11_train_all.yaml
log='training'

python kitti_submission.py --config-file ./configs/kitti2015-multi.yaml\
    --checkpoint-path $root/epoch=004.ckpt\
    --data-root /home/faleotti/projects/data/KITTI-Multiview/KITTI-2015/\
    --annfile $ann\
    --resize-to-shape 384 1284\
    --log_dir $root/004/$log
