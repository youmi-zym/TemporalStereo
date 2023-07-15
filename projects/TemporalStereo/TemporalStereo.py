import time

import ipdb
import pandas as pd
import math
from typing import Union, List, Dict, Tuple, Optional
import random
EXPMAX = 50


import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import CfgNode

from architecture.data.datasets import build_stereo_dataset
from architecture.data.evaluation import do_evaluation, do_occlusion_evaluation
from architecture.modeling import build_backbone, build_aggregation, DispSmoothL1Loss, WarssersteinDistanceLoss
from architecture.modeling.layers import project_to_3d, FunctionSoftsplat
from architecture.utils import disp_to_color, colormap, disp_err_to_colorbar


class TemporalStereo(pl.LightningModule):
    """ Network architecture implementing TemporalStereo"""

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.cfg = CfgNode(hparams)
        cfg = self.cfg
        self.frame_idxs = cfg.FRAME_IDXS
        self.max_disp = cfg.MAX_DISP

        self.with_previous = cfg.MODEL.WITH_PREVIOUS
        self.previous_with_gradient = cfg.MODEL.PREVIOUS_WITH_GRADIENT

        self.local_map_size = cfg.MODEL.LOCAL_MAP_SIZE
        self.use_past_cost = cfg.MODEL.USE_PAST_COST

        self.backbone = build_backbone(cfg)
        self.aggregation = build_aggregation(cfg)
        self.smooth_l1_loss = DispSmoothL1Loss(cfg.MODEL.LOSSES.SMOOTH_L1_LOSS)
        self.warsserstein_distance_loss = WarssersteinDistanceLoss(cfg.MODEL.LOSSES.WARSSERSTEIN_DISTANCE_LOSS)

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                m.eval()

    def _assert_bn_freezed(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                assert not m.training, m.training

    def train_dataloader(self):
        dataset = build_stereo_dataset(self.cfg.DATA.TRAIN, 'train')

        self.train_dataset_length = len(dataset)
        self.logger.filewriter.set_num_total_steps(len(dataset) // self.cfg.DATA.TRAIN.BATCH_SIZE * self.cfg.TRAINER.MAX_EPOCHS)
        self.logger.filewriter.set_start_time(time.time())

        dataloader = DataLoader(
            dataset, self.cfg.DATA.TRAIN.BATCH_SIZE, shuffle=True,
            num_workers=self.cfg.DATA.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=False)

        return dataloader

    def val_dataloader(self):
        dataset = build_stereo_dataset(self.cfg.DATA.VAL, 'val')

        self.val_dataset_length = len(dataset)
        dataloader = DataLoader(
            dataset, self.cfg.DATA.VAL.BATCH_SIZE, shuffle=False,
            num_workers=self.cfg.DATA.VAL.NUM_WORKERS, pin_memory=True, drop_last=False)

        return dataloader

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        parameters_to_train = []
        parameters_to_train += list(self.parameters())

        # optimzer
        if self.cfg.OPTIMIZER.TYPE == 'Adam':
            lr = self.cfg.OPTIMIZER.ADAM.LR
            betas = self.cfg.OPTIMIZER.ADAM.get('BETAS', (0.9, 0.999))
            optimizer = torch.optim.Adam([
                {'params': parameters_to_train, 'lr': lr, 'betas':betas}])
            optimizers.append(optimizer)
        elif self.cfg.OPTIMIZER.TYPE == 'RMSProp':
            lr = self.cfg.OPTIMIZER.RMSPROP.LR
            optimizer = torch.optim.RMSprop([
                {'params': parameters_to_train, 'lr': lr}])
            optimizers.append(optimizer)
        elif self.cfg.OPTIMIZER.TYPE == 'AdamW':
            lr = self.cfg.OPTIMIZER.ADAMW.LR
            betas = self.cfg.OPTIMIZER.ADAMW.get('BETAS', (0.9, 0.999))
            weight_decay = self.cfg.OPTIMIZER.ADAMW.get('WEIGHT_DECAY', 1e-4)
            optimizer = torch.optim.AdamW([
                {'params': parameters_to_train, 'lr': lr, 'betas': betas, 'weight_decay': weight_decay}])
            optimizers.append(optimizer)

        else:
            raise NotImplementedError(
                'optimizer %s not supported' % self.cfg.OPTIMIZER.TYPE)

        # scheduler
        if self.cfg.SCHEDULER.TYPE == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.cfg.SCHEDULER.STEP_LR.STEP_SIZE,
                gamma=self.cfg.SCHEDULER.STEP_LR.GAMMA)
            schedulers.append(scheduler)

        elif self.cfg.SCHEDULER.TYPE == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.cfg.SCHEDULER.MULTI_STEP_LR.MILESTONES,
                gamma=self.cfg.SCHEDULER.MULTI_STEP_LR.GAMMA)
            schedulers.append(scheduler)

        elif self.cfg.SCHEDULER.TYPE == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.cfg.SCHEDULER.EXPONENTIAL_LR.GAMMA)
            schedulers.append(scheduler)

        elif self.cfg.SCHEDULER.TYPE != 'None':
            raise NotImplementedError(
                'scheduler %s not supported' % self.cfg.SCHEDULER.TYPE)

        return optimizers, schedulers

    def on_train_epoch_start(self) -> None:
        # self._freeze_bn()
        pass

    def training_step(self, batch, batch_idx):
        # self._assert_bn_freezed()
        losses = {}
        before_op_time = time.time()
        outputs = self.multi_frame_forward(batch, is_train=True)
        for frame_idx in self.frame_idxs:
            if not self.previous_with_gradient and frame_idx != 0:
                continue

            l1_loss_dict = self.smooth_l1_loss(outputs[('disps', frame_idx, 'l')], batch[('disp_gt', frame_idx, 'l')])
            for key in l1_loss_dict.keys():
                value = l1_loss_dict[key]
                losses['{}_'.format(frame_idx) + key] = value

            war_loss_dict = self.warsserstein_distance_loss(outputs[('costs', frame_idx, 'l')],
                                                            outputs[('offsets', frame_idx, 'l')],
                                                            outputs[('disp_samples', frame_idx, 'l')],
                                                            batch[('disp_gt', frame_idx, 'l')])
            for key in war_loss_dict.keys():
                value = war_loss_dict[key]
                losses['{}_'.format(frame_idx) + key] = value

        total_loss = 0
        for key in losses.keys():
            if isinstance(key, str) and key.find('loss') > -1:
                total_loss += losses[key]
        losses["loss"] = total_loss

        duration = time.time() - before_op_time
        self.log_dict(losses, logger=True, on_step=True)

        if self.global_step % self.cfg.TRAINER.FLUSH_LOGS_EVERY_N_STEPS == 0:
            self.logger.filewriter.log_time(self.global_step, self.current_epoch, batch_idx,
                                            self.cfg.DATA.TRAIN.BATCH_SIZE, duration, losses["loss"])
        # randomly visulize some batch
        if self.global_step % 2000 == 0:
            self.log_image(self.cfg.DATA.VAL.BATCH_SIZE, batch, outputs, prefix='train_')

        return {"loss": losses['loss']}

    def validation_step(self, batch, batch_idx):
        outputs = self.multi_frame_forward(batch, is_train=False)

        # remove padding
        gh, gw = batch[('disp_gt', 0, 'l')].shape[-2:]

        if ('disp_warp', 0, 'l') in outputs:
            num = len(outputs[('disp_warp', 0, 'l')])
            for i in range(num):
                outputs[('disps', 0, 'l')].append(outputs[('disp_warp', 0, 'l')][i])
        if ('disp_warp_gt', 0, 'l') in outputs:
            outputs[('disps', 0, 'l')].append(outputs[('disp_warp_gt', 0, 'l')])

        outputs[('disps', 0, 'l')] = [F.interpolate(disp * gw / disp.shape[-1], size=(gh, gw), mode='bilinear', align_corners=True) for disp in outputs[('disps', 0, 'l')]]

        whole_error_dict = self.log_metric(batch, outputs)

        self.log_dict(whole_error_dict, logger=True, on_epoch=True, reduce_fx=torch.mean)

        # randomly visulize some batch
        if batch_idx == (self.current_epoch % (self.val_dataset_length // self.cfg.VAL.VIS_INTERVAL)):
            self.log_image(self.cfg.DATA.VAL.BATCH_SIZE, batch, outputs)

        return whole_error_dict

    def validation_epoch_end(self, outputs) -> None:
        self.logger.filewriter.stdout("\n\n" + "*"*40 + "  Validation on Epoch: {}  ".format(self.current_epoch) + "*"*40 + "\n")
        self.process_error_dict(outputs)

    def test_step(self, batch, batch_idx):
        outputs = self.multi_frame_forward(batch, is_train=False)

        # remove padding
        gh, gw = batch[('disp_gt', 0, 'l')].shape[-2:]
        outputs[('disps', 0, 'l')] = [F.interpolate(disp * gw / disp.shape[-1], size=(gh, gw), mode='bilinear', align_corners=True) for disp in outputs[('disps', 0, 'l')]]

        whole_error_dict = self.log_metric(batch, outputs)

        self.log_dict(whole_error_dict, logger=True, on_epoch=True, reduce_fx=torch.mean)

        return whole_error_dict

    def test_epoch_end(self, outputs) -> None:
        self.logger.filewriter.stdout("\n\n" + "*"*40 + "  Final Test  " + "*"*40 + "\n")
        self.process_error_dict(outputs)

    @rank_zero_only
    def process_error_dict(self, outputs) -> None:
        keys = outputs[0].keys()
        # output evaluation result in output_dict
        # for each disparity/flow map, the result will contain one pd.DataFrame, region as index, metric as columns
        output_dict = {}
        for key in keys:
            val = torch.stack([torch.Tensor([out[key]])for out in outputs if out is not None]).mean().cpu().numpy()
            val = "{:.4f}".format(float(val))
            _include_keys = ['metric_disparity', 'metric_backwardflow', 'metric_forwardflow']
            for inc_key in _include_keys:
                if inc_key in key:  # e.g. 'metric_disparity_0/all_epe', 'metric_backwardflow_0/all_epe'
                    map_id = key.split('/')[0]
                    region, metric = key.split('/')[1].split('_')

                    # each map contains one pd.DataFrame, area as index, metric as columns
                    if map_id not in output_dict.keys():
                        output_dict[map_id] = {}
                    if region not in output_dict[map_id].keys():
                        output_dict[map_id][region] = {}
                    output_dict[map_id][region][metric] = val
                else:
                    pass
                    # output_dict[key] = val

        # generate pandas
        info = "\n"
        for key in list(output_dict):
            output_dict[key] = pd.DataFrame.from_dict(output_dict[key], orient='index')
            info += "{}\n".format(key)
            info += "{}\n".format(output_dict[key])

        self.logger.filewriter.stdout(info)

    def multi_frame_forward(self, batch, is_train=False):
        final_outputs = {}
        outputs = {}
        self.frame_idxs.sort()
        for i, timestamp in enumerate(self.frame_idxs):
            self.current_timestamp = timestamp

            if i == 0 and self.with_previous:
                outputs[('prev_info', timestamp-1, 'l')] = {}

            if self.previous_with_gradient:
                assert self.with_previous
                outs = self(batch, outputs, is_train=is_train, timestamp=timestamp)
                final_outputs.update(outs)
            else:
                if timestamp == 0:
                    outs = self(batch, outputs, is_train=is_train, timestamp=timestamp)
                    final_outputs.update(outs)
                else:
                    reset_to_training = self.training
                    with torch.no_grad():
                        self.eval()
                        outs = self(batch, outputs, is_train=False, timestamp=timestamp)
                        if reset_to_training:
                            self.train()

            outputs = {}
            if self.with_previous:
                outputs[('prev_info', timestamp, 'l')] = outs[('prev_info', timestamp, 'l')]

        return final_outputs

    def forward(self, batch, outputs, is_train=False, timestamp=0):
        """
        Pass a minibatch through the network and generate images and losses
        """
        bs, c, full_h, full_w = batch[('color_aug', timestamp, 'l')].shape
        prev_info = outputs.get(('prev_info', timestamp-1, 'l'), {})
        left_image, right_image = batch[('color_aug', timestamp, 'l')], batch[('color_aug', timestamp, 'r')]

        left_feats, right_feats, prev_info = self.backbone(left_image, right_image, prev_info)

        if self.with_previous and ((timestamp-1) in self.frame_idxs):
            outs, prev_info = self.update_map(batch, prev_info, timestamp)
            outputs.update(outs)
        else:
            prev_info.pop('cost_memory', None)
            prev_info.pop('use_past_cost', False)
            prev_info.pop('prev_disp', None)
            prev_info.pop('local_map', None)
            prev_info.pop('local_map_size', 0)

        estdisps, costs, disp_samples, offsets, search_ranges, prev_info = self.aggregation(left_feats, right_feats,
                                                                                            left_image, right_image,
                                                                                            prev_info=prev_info)
        disps = []
        for i, d in enumerate(estdisps):
            dh, dw = d.shape[-2:]
            inter_disp = F.interpolate(d * full_w / dw, size=(full_h, full_w), mode='bilinear', align_corners=True)
            disps.append(inter_disp)

        # finally
        prev_info['left_feats'] = left_feats
        prev_info['right_feats'] = right_feats
        outputs[('left_feats', timestamp, 'l')] = left_feats
        outputs[('right_feats', timestamp, 'l')] = right_feats

        outputs[('disps', timestamp, 'l')] = disps
        outputs[('costs', timestamp, 'l')] = costs
        outputs[('offsets', timestamp, 'l')] = offsets
        outputs[('disp_samples', timestamp, 'l')] = disp_samples
        outputs[('search_ranges', timestamp, 'l')] = search_ranges
        outputs[('prev_info', timestamp, 'l')] = prev_info

        return outputs

    def update_map(self, batch, prev_info, timestamp):
        outputs = {}
        baseline = batch['baseline'].float()
        full_h, full_w = batch[('color_aug', timestamp, 'l')].shape[-2:]

        K = batch[('K', 0)]
        # get extrinsic
        T_past_to_now = prev_info.get('T_past_to_now', None)
        if T_past_to_now is None:
            past_inv_T = batch[('inv_T', timestamp - 1, 'l')]
            now_T = batch[('T', timestamp, 'l')]
            T_past_to_now = torch.bmm(now_T, past_inv_T)
            outputs[('T_past_to_now_gt', timestamp, 'l')] = T_past_to_now

        def update_local_map(prev_disp, local_map=None):
            if local_map is not None:
                height, width = local_map.shape[-2:]
            else:
                height, width = full_h//8, full_w//8
            downscale_factor = full_w / width
            # get intrinsic
            down_K = torch.cat((
                K[:, 0:1, :] / downscale_factor,
                K[:, 1:2, :] / downscale_factor,
                K[:, 2:, :],
            ), dim=1)
            down_inv_K = torch.inverse(down_K)
            focal_length = down_K[:, 0, 0].view(-1, 1, 1, 1)

            # get optical flow
            prev_disp = F.interpolate(prev_disp*width/prev_disp.shape[-1], size=(height, width), mode='bilinear', align_corners=True)
            prev_depth = baseline * focal_length / (prev_disp + 1e-5)
            out_project = project_to_3d(prev_depth, down_K, down_inv_K, T_past_to_now)
            forward_flow = out_project['optical_flow'][:, :2, :, :]
            updated_prev_depth = out_project['triangular_depth']
            updated_prev_disp = baseline * focal_length / (updated_prev_depth + 1e-5)
            warp_disp = FunctionSoftsplat(tenInput=updated_prev_disp,
                                          tenFlow=forward_flow,
                                          tenMetric=(prev_disp[:, :1] - prev_disp[:, :1].mean()).clamp(-50, 50),
                                          strType='softmax')

            if local_map is None:
                local_map = warp_disp
            else:
                local_map = torch.cat([prev_disp, local_map], dim=1)
                if local_map.shape[1] > self.local_map_size:
                    local_map = local_map[:, :self.local_map_size]
                local_depth = baseline * focal_length / (local_map + 1e-5)
                local_out_project = project_to_3d(local_depth, down_K, down_inv_K, T_past_to_now)
                local_forward_flow = local_out_project['optical_flow'][:, :2, :, :]
                updated_local_depth = local_out_project['triangular_depth']
                updated_local_map = baseline * focal_length / (updated_local_depth + 1e-5)
                local_map = FunctionSoftsplat(tenInput=updated_local_map,
                                              tenFlow=local_forward_flow,
                                              tenMetric=(prev_disp[:, :1] - prev_disp[:, :1].mean()).clamp(-EXPMAX, EXPMAX), # avoid explosion when exp()
                                              strType='softmax')
            local_map = local_map.detach()

            return local_map

        def update_past_cost(prev_disp, memory):
            # get memory
            disp_sample = memory['disp_sample'].detach()
            cost_volume = memory['cost_volume'].detach()
            sample_b, sample_c, sample_h, sample_w = disp_sample.shape

            # get previous disparity map and intrinsics
            downscale_factor = full_w / sample_w
            # get intrinsic
            down_K = torch.cat((
                K[:, 0:1, :] / downscale_factor,
                K[:, 1:2, :] / downscale_factor,
                K[:, 2:, :],
            ), dim=1)
            down_inv_K = torch.inverse(down_K)
            focal_length = down_K[:, 0, 0].view(-1, 1, 1, 1)

            # get optical flow
            prev_disp = F.interpolate(prev_disp*sample_w/prev_disp.shape[-1], size=(sample_h, sample_w), mode='bilinear', align_corners=True)
            prev_depth = baseline * focal_length / (prev_disp + 1e-5)
            out_project = project_to_3d(prev_depth, down_K, down_inv_K, T_past_to_now)
            forward_flow = out_project['optical_flow'][:, :2, :, :]

            # project prev disparity samples to current
            depth_sample = baseline * focal_length / (disp_sample + 1e-5)
            out_project = project_to_3d(depth_sample, down_K, down_inv_K, T_past_to_now)
            updated_depth_sample = out_project['triangular_depth']
            updated_disp_sample = baseline * focal_length / (updated_depth_sample + 1e-5)

            sample_cost = torch.cat([updated_disp_sample, cost_volume], dim=1)
            warp_sample_cost = FunctionSoftsplat(tenInput=sample_cost,
                                                 tenFlow=forward_flow,
                                                 tenMetric=(prev_disp[:, :1] - prev_disp[:, :1].mean()).clamp(-EXPMAX, EXPMAX), # avoid explosion when exp()
                                                 strType='softmax')

            memory = {
                'disp_sample': warp_sample_cost[:, :sample_c, :, :].detach(),
                'cost_volume': warp_sample_cost[:, sample_c:, :, :].detach(),
            }

            return memory

        # move depth in previous camera to current
        vis_local_map = []
        if self.with_previous:
            prev_disp = prev_info['prev_disp'].detach()

            cost_memory = prev_info.get('cost_memory', None)
            if self.use_past_cost and cost_memory is not None:
                cost_memory = update_past_cost(prev_disp, memory=cost_memory)

                vis_local_map.append(cost_memory['disp_sample'])
            elif not self.use_past_cost:
                cost_memory = None

            prev_info['cost_memory'] = cost_memory
            prev_info['use_past_cost'] = self.use_past_cost

            if self.local_map_size > 0:
                local_map = prev_info.get('local_map', None)
                local_map = update_local_map(prev_disp, local_map=local_map)
                prev_info['local_map'] = local_map
                prev_info['local_map_size'] = self.local_map_size

                # for visualization
                vh, vw = local_map.shape[-2:]
                vis_local_map.append(local_map)
                vis_local_map.append(F.interpolate(prev_disp * vw/ prev_disp.shape[-1],
                                                   size = (vh, vw),
                                                   mode='bilinear',
                                                   align_corners=True))

            if len(vis_local_map) > 0:
                outputs[('local_map', timestamp, 'l')] = torch.cat(vis_local_map, dim=1)

        return outputs, prev_info

    def log_metric(self, batch, outputs):
        whole_error_dict = {}

        if ('disp_gt', 0, 'l') in batch:
            disps = outputs[('disps', 0, 'l')]
            eval_disparity_ids = self.cfg.VAL.get('EVAL_DISPARITY_IDS', [i for i in range(len(disps))])
            eval_disparity_ids = [id for id in eval_disparity_ids if id < len(disps)]
            for id in eval_disparity_ids:
                all_error_dict = do_evaluation(
                    disps[id], batch[('disp_gt', 0, 'l')], self.cfg.VAL.LOWERBOUND, self.cfg.VAL.UPPERBOUND)

                for key in all_error_dict.keys():
                    whole_error_dict['metric_disparity_{}/all_'.format(id) + key] = all_error_dict[key]

                if self.cfg.VAL.DO_OCCLUSION_EVALUATION and (batch[('disp_gt', 0, 'l')] is not None) and (batch[('disp_gt', 0, 'r')] is not None):

                    noc_occ_error_dict = do_occlusion_evaluation(
                        disps[id], batch[('disp_gt', 0, 'l')], batch[('disp_gt', 0, 'r')],
                        self.cfg.VAL.LOWERBOUND, self.cfg.VAL.UPPERBOUND)

                    for key in noc_occ_error_dict.keys():
                        whole_error_dict['metric_disparity_{}/'.format(id) + key] = noc_occ_error_dict[key]

        return whole_error_dict

    @rank_zero_only
    def log_image(self, batch_size, inputs, outputs, prefix=''):
        """Write an event to the tensorboard events file
        """
        h, w = inputs[('color', 0, 'l')].shape[-2:]
        full_res = (h, w)
        if self.training:
            step = self.global_step
        else:
            step = self.current_epoch

        for bs in range(min(4, batch_size)):  # write a maxmimum of four images
            for frame_id in self.frame_idxs:
                self.logger.experiment.add_image(
                    prefix+"color_{}_{}/{}".format(frame_id, 'l', bs),
                    inputs[("color", frame_id, 'l')][bs].data,
                    step,
                )
                self.logger.experiment.add_image(
                    prefix+"color_{}_{}/{}".format(frame_id, 'r', bs),
                    inputs[("color", frame_id, 'r')][bs].data,
                    step,
                )

            max_disp = None
            disp_gt = None
            if ('disp_gt', 0, 'l') in inputs:
                disp_gt = inputs[('disp_gt', 0, 'l')][bs, 0].cpu().numpy()
                max_disp = disp_gt.max()
                disp_gt_color = colormap(disp_to_color, disp_gt, normalize=False, format='CHW', max_disp=max_disp)
                self.logger.experiment.add_image(
                    prefix+"disparity_gt_0/{}".format(bs),
                    disp_gt_color, step)

            def vis_disp_and_error(bs, disp, disp_gt, disp_key, error_key, max_disp):
                disp_color = colormap(disp_to_color, disp[bs, 0].detach().cpu().numpy(), normalize=False, format='CHW', max_disp=max_disp)
                self.logger.experiment.add_image(disp_key, disp_color, step)

                if disp_gt is not None:
                    gh, gw = disp_gt.shape[-2:]
                    wh, ww = disp.shape[-2:]
                    # (w, h) in inch
                    if gw > gh:
                        figsize = (int(gw/gh*6), 6)
                    else:
                        figsize = (6, int(gh/gw*6))
                    figure = plt.figure(figsize=figsize, dpi=80, tight_layout=True)
                    disp = F.interpolate(disp*gw/ww, size=(gh, gw), mode='bilinear', align_corners=True)
                    disp = disp[bs, 0].detach().cpu().numpy()
                    error_color = colormap(disp_err_to_colorbar, disp, disp_gt, normalize=False, format='HWC', with_bar=True)
                    plt.imshow(error_color)
                    self.logger.experiment.add_figure(error_key, figure, step)
                    plt.close(figure)


            if ('local_map', 0, 'l') in outputs:
                local_map = outputs[('local_map', 0, 'l')]
                local_disp_gt = None
                if disp_gt is not None:
                    gh, gw = disp_gt.shape[-2:]
                    mh, mw = local_map.shape[-2:]
                    local_map = F.interpolate(local_map*gw/mw, size=(gh, gw), mode='bilinear', align_corners=True)
                    b, c, h, w = local_map.shape
                    local_map = local_map.reshape(b, 1, c*h, w).contiguous()
                    local_disp_gt = torch.from_numpy(disp_gt).repeat(c,1).cpu().numpy()

                vis_disp_and_error(bs, local_map, local_disp_gt,
                                   disp_key=prefix + "local_map/{}".format(bs),
                                   error_key=prefix + "local_map_errorbar/{}".format(bs),
                                   max_disp=max_disp)

            for disp_id, disp in enumerate(outputs[('disps', 0, 'l')]):
                vis_disp_and_error(bs, disp, disp_gt,
                                   disp_key=prefix + "disparity_{}/{}".format(disp_id, bs),
                                   error_key=prefix + "disparity_errorbar_{}/{}".format(disp_id, bs),
                                   max_disp=max_disp)


            search_ranges = outputs.get(('search_ranges', 0, 'l'), None)
            if search_ranges is not None:
                full_h, full_w = full_res
                for i, sr in enumerate(search_ranges):
                    low = sr['low']
                    high = sr['high']
                    h, w = low.shape[-2:]
                    lvl = int(math.log2(full_w/w))
                    low = F.interpolate(low*full_w/w, size=(full_h, full_w), mode='bilinear', align_corners=True)
                    high = F.interpolate(high*full_w/w, size=(full_h, full_w), mode='bilinear', align_corners=True)
                    vis_disp_and_error(bs, low, disp_gt,
                                       disp_key=prefix + "low_disparity_{}/{}".format(lvl, bs),
                                       error_key=prefix + "low_errorbar_{}/{}".format(lvl, bs),
                                       max_disp=max_disp)
                    vis_disp_and_error(bs, high, disp_gt,
                                       disp_key=prefix + "high_disparity_{}/{}".format(lvl, bs),
                                       error_key=prefix + "high_errorbar_{}/{}".format(lvl, bs),
                                       max_disp=max_disp)

                    if disp_gt is not None:
                        mask = (disp_gt > 0) & (disp_gt < self.max_disp)
                        gh, gw = disp_gt.shape[-2:]
                        h, w = low.shape[-2:]
                        low = F.interpolate(low * gw / w, size=(gh, gw), mode='bilinear', align_corners=True)
                        high = F.interpolate(high * gw / w, size=(gh, gw), mode='bilinear', align_corners=True)
                        low = low[bs, 0].detach().cpu().numpy()
                        high = high[bs, 0].detach().cpu().numpy()
                        valid = mask & (low <= disp_gt) & (high >= disp_gt)
                        margin = (high - low) * valid * sr['low'].shape[-1] / gw
                        margin_gt = (margin - margin)+1e-3
                        if gw > gh:
                            figsize = (int(gw / gh * 6), 6)
                        else:
                            figsize = (6, int(gh / gw * 6))
                        figure = plt.figure(figsize=figsize, dpi=80, tight_layout=True)
                        margin_color = colormap(disp_err_to_colorbar, margin, margin_gt, normalize=False, format='HWC', with_bar=True)
                        margin_color[:gh, :gw, :] = margin_color[:gh, :gw, :] * (valid[:gh, :gw, None] * 1.0)
                        plt.imshow(margin_color)
                        self.logger.experiment.add_figure(prefix+'search_range_{}/{}'.format(lvl, bs), figure, step)
                        plt.close(figure)
                        valid = valid | (~mask)
                        valid_color = colormap('gray', valid * 1.0, normalize=False, format='CHW')
                        self.logger.experiment.add_image(
                            prefix + 'search_range_valid_{}/{}'.format(lvl, bs),
                            valid_color, step
                        )

            for idx in self.frame_idxs:
                if idx == 0:
                    continue
                if ('disp_gt', idx, 'l') in inputs:
                    disp_gt_prev = inputs[('disp_gt', idx, 'l')][bs, 0].cpu().numpy()
                    disp_gt_prev_color = colormap(disp_to_color, disp_gt_prev, normalize=False, format='CHW', max_disp=max_disp)
                    self.logger.experiment.add_image(
                        prefix+"disparity_gt_{}/{}".format(idx, bs),
                        disp_gt_prev_color, step)

