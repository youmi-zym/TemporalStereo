# Copyright CVLAB of University of Bologna 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the StereoBenchmark licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.utilities import rank_zero_only

seed_everything(43, workers=True)

import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader

import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

from config import get_cfg, get_parser
from TemporalStereo import TemporalStereo
from architecture.data.datasets import build_stereo_dataset
from logger import Logger

import shutil

@rank_zero_only
def backup_code(save_dir):
    savedir = '{}/code/'.format(save_dir)
    datadirs = ['architecture',
                'projects',
                ]
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir)
    root = osp.join(osp.dirname(osp.abspath(__file__)), '../../')
    for datadir in datadirs:
        shutil.copytree('{}{}'.format(root, datadir),
                        '{}{}'.format(savedir, datadir),
                        ignore=shutil.ignore_patterns('*.pyc', '*.npy', '*.pdf', '*.json', '*.bin', '.idea',
                                                      '*.egg', '*.egg-info', 'build', 'dist', '*.so',
                                                      '*.pth', '*.pkl', '*.ckpt',
                                                      '__pycache__', '.DS_Store', ))

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = get_cfg(args)
    model = TemporalStereo(cfg.convert_to_dict())

    save_path = os.path.join(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION)
    logger = Logger(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch:03d}' if cfg.CHECKPOINT.EVERY_N_EPOCHS > 0 else '{epoch}-{step}',
        dirpath=save_path,
        monitor=None,
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=cfg.CHECKPOINT.EVERY_N_TRAIN_STEPS,
        every_n_epochs=cfg.CHECKPOINT.EVERY_N_EPOCHS,)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    swa = StochasticWeightAveraging(swa_epoch_start=0.8)

    if os.path.isfile(cfg.TRAINER.LOAD_FROM_CHECKPOINT):
        checkpoint = torch.load(cfg.TRAINER.LOAD_FROM_CHECKPOINT)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.filewriter.stdout("Load checkpoint from: {}!".format(cfg.TRAINER.LOAD_FROM_CHECKPOINT))
    elif len(cfg.TRAINER.LOAD_FROM_CHECKPOINT) > 0:
        logger.filewriter.stdout("Warning: Checkpoint at {} doesn't exist!".format(cfg.TRAINER.LOAD_FROM_CHECKPOINT))


    # backup code
    backup_code(save_dir=save_path)

    trainer = pl.Trainer(
        logger=logger,
        strategy='ddp',
        benchmark=True,  # speed up training, about 2x
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, lr_monitor, swa],
        check_val_every_n_epoch=cfg.TRAINER.CHECK_VAL_EVERY_N_EPOCHS,
        resume_from_checkpoint=cfg.TRAINER.RESUME_FROM_CHECKPOINT if os.path.isfile(cfg.TRAINER.RESUME_FROM_CHECKPOINT) else None,
        gpus=cfg.TRAINER.NUM_GPUS,
        num_nodes=cfg.TRAINER.NUM_NODES,
        max_epochs=cfg.TRAINER.MAX_EPOCHS,
        precision=cfg.TRAINER.PRECISION,
        amp_backend='native',
        sync_batchnorm=cfg.TRAINER.SYNC_BATCHNORM,
        detect_anomaly=True,
        gradient_clip_val= cfg.TRAINER.GRADIENT_CLIP_VAL,
        accumulate_grad_batches=1,
        fast_dev_run=False,
        # limit_train_batches=0.002, limit_val_batches=0.01, limit_test_batches=0.005,
    )

    # ----------------------------------------- Train ----------------------------------------- #

    trainer.fit(model)

    # ----------------------------------------- Test  ----------------------------------------- #
    dataset = build_stereo_dataset(cfg.DATA.TEST, 'test')
    dataloader = DataLoader(
        dataset, cfg.DATA.TEST.BATCH_SIZE, shuffle=False,
        num_workers=cfg.DATA.TEST.NUM_WORKERS, pin_memory=True, drop_last=False)

    trainer.test(model,
                 test_dataloaders=dataloader,
                 ckpt_path=os.path.join(save_path, 'epoch={:03d}.ckpt'.format(cfg.TRAINER.MAX_EPOCHS-1)))

    print("Done!")