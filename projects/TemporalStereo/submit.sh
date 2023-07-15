#!/bin/bash

CONFIG=./configs/kitti2015-multi.yaml
EXP_ROOT=./exps/TemporalStereo/kitti2015/multi
DATA_ROOT=./datasets/KITTI-Multiview/KITTI-2015/
CKPT=$EXP_ROOT/ckpt_best.ckpt
ANN=./splits/view_11_train_all.yaml

python kitti_submission.py --config-file $CONFIG \
    --checkpoint-path $CKPT \
    --data-root $DATA_ROOT \
    --annfile $ANN \
    --resize-to-shape 384 1284 \
    --log_dir $EXP_ROOT/output
