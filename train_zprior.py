# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import shutil
import sys
import time
import uuid
from collections import defaultdict
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import gym
import hydra
import numpy as np
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from fairmotion.data import amass
from omegaconf import DictConfig, OmegaConf
from pytorch3d import transforms as p3t
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

import hucc
from hucc import mocap as hm
from hucc.mocap import (OfflineMocapFrameDataset, OfflineMocapSequenceDataset,
                        OfflineMocapStackDataset,
                        OfflineMocapSubSequenceDataset, ReplayEnv)
from hucc.mocap.datasets import comic_reward, th_R2R6, th_R62R
from hucc.spaces import th_flatten, th_unflatten

log = logging.getLogger(__name__)


def postproc_comic_loss(inp, outp, umask):
    inp = copy(inp)
    outp = copy(outp)

    # accumulate with quaternion multiply for relative root orientations
    if 'rrelr6' in inp and 'br6' in inp and not 'r6' in inp:
        for dc in (inp, outp):
            dq = p3t.matrix_to_quaternion(th_R62R(dc['rrelr6'])).transpose(0, 1)
            rq = th.stack(list(accumulate(dq, p3t.quaternion_multiply)), dim=1)
            rr6 = th_R2R6(p3t.quaternion_to_matrix(rq))
            dc['r6'] = th.cat([rr6, dc['br6']], dim=-1)
            dc['_rq'] = rq
    elif 'rquatr' in inp and 'br6' in inp and not 'r6' in inp:
        for dc in (inp, outp):
            dq = dc['rquatr'].transpose(0, 1)
            rq = th.stack(list(accumulate(dq, p3t.quaternion_multiply)), dim=1)
            rr6 = th_R2R6(p3t.quaternion_to_matrix(rq))
            dc['r6'] = th.cat([rr6, dc['br6']], dim=-1)
            dc['_rq'] = rq
    elif 'rquatr' in inp and 'bquat' in inp and not 'quat' in inp:
        for dc in (inp, outp):
            dq = dc['rquatr'].transpose(0, 1)
            rq = th.stack(list(accumulate(dq, p3t.quaternion_multiply)), dim=1)
            dc['quat'] = th.cat([rq, dc['bquat']], dim=-1)
            dc['_rq'] = rq

    # cumsum for delta positions over time dimension, data is BxTxC
    if 'relpos' in inp and not 'rpos' in inp:
        for dc in (inp, outp):
            dc['rpos'] = th.cumsum(dc['relpos'], dim=1)
    elif 'relxypos' in inp and not 'rpos' in inp:
        for dc in (inp, outp):
            dc['rpos'] = dc['relxypos'].clone()
            scale = 1
            dc['rpos'].narrow(-1, 0, 2).copy_(
                th.cumsum(dc['relxypos'].narrow(-1, 0, 2).div(scale), dim=1)
            )
    elif 'rlvel' in inp and not 'rpos' in inp:
        for dc in (inp, outp):
            rq = dc.get('_rq', None)
            if rq is None:
                rq = p3t.matrix_to_quaternion(th_R62R(dc['r6']))
            rqi = p3t.quaternion_invert(rq)
            rvel = p3t.quaternion_apply(rqi, dc['rlvel'])
            dc['rpos'] = th.cumsum(rvel, dim=1)

    inpm, outpm = {}, {}
    shapes = {
        'rpos': (-1, 3),
        'rvel': (-1, 3),
        'relxypos': (-1, 3),
        'r6': (-1, 22, 6),
        'avel': (-1, 22, 3),
        'lvel': (-1, 22, 3),
        'quat': (-1, 22, 4),
    }
    for key, shape in shapes.items():
        if key in inp:
            inpm[key] = inp[key].masked_select(umask).view(shape)
            outpm[key] = outp[key].masked_select(umask).view(shape)
    return inpm, outpm


def compute_loss(d, inp, mask, outs, mloss, kl):
    model_acc = d.model
    if dist.is_initialized():
        model_acc = d.model.module

    losses = []
    metrics = {}
    if d.cfg.loss.startswith('comic'):
        umask = mask.unsqueeze(-1)
        for i in range(len(outs)):
            outp = th_unflatten(d.obs_space, outs[i])
            inpm, outpm = postproc_comic_loss(inp, outp, umask)
            crm = comic_reward(inpm, outpm, d.skel, d.xform_from_parent_joint)
            if d.cfg.loss.startswith('comic_'):
                ls = []
                for lpart in d.cfg.loss.split('_')[1:]:
                    ls.append(crm[f'diff_{lpart}'])
                losses.append(th.cat(losses, dim=-1).mean())
            else:
                losses.append(-crm['comic_reward'].mean())
            metrics.update(**{f'{k}_l{i+1}': v for k, v in crm.items()})
        losses.append(d.kl_reg * kl)
    elif d.cfg.loss.startswith('l2_'):
        umask = mask.unsqueeze(-1)
        for i in range(len(outs)):
            outp = th_unflatten(d.obs_space, outs[i])
            inpm, outpm = postproc_comic_loss(inp, outp, umask)
            keys = d.cfg.loss.split('_')[1:]
            loss = None
            for k in keys:
                tloss = (inpm[k] - outpm[k]).square().mean()
                loss = loss + tloss if loss is not None else tloss
                metrics[f'{k}_l{i+1}'] = tloss
            losses.append(loss)
        losses.append(d.kl_reg * kl)
    elif d.prior:
        B = mask.shape[0]
        umask = mask
        if mloss.ndim == 3:
            losses.append(th.sum(mloss.mean(dim=-1) * umask) / mask.sum())
        else:
            losses.append(th.sum(mloss.view(B, -1) * umask) / mask.sum())
    elif mloss is not None:
        losses.append(th.sum(mloss * mask.unsqueeze(-1)) / mask.sum())

    loss = sum(losses)
    metrics['loss'] = loss.detach()
    return loss, metrics


def eval(d, updates, env):
    device = d.cfg.device
    in_keys = d.cfg.dataset.inputs
    fwd_loss = d.cfg.loss
    if (
        fwd_loss.startswith('comic')
        or fwd_loss.startswith('l2_')
        or fwd_loss in ('hmvae', 'quat', 'smpl')
    ):
        fwd_loss = 'l2'
    model = d.model
    model_acc = model
    if dist.is_initialized():
        model_acc = model.module

    eval_mloss = []
    eval_metrics = defaultdict(list)
    eval_nsmpl = 0
    eval_batch = 0

    # Evaluate next-step prediction
    for inp, lbl, mask in d.dl['valid']:
        lbl = lbl.to(device, non_blocking=True)
        inp = {k: v.to(device, non_blocking=True) for k, v in inp.items()}
        inp = {k: v.squeeze() for k, v in inp.items()}
        if 'rpos' in inp:
            xy_base = inp['rpos'].narrow(1, 0, 1).narrow(2, 0, 2).clone()
            inp['rpos'].narrow(2, 0, 2).sub_(xy_base)
        if 'relxypos' in inp:
            inp['relxypos'].narrow(2, 0, 2)
        mask = mask.to(device, non_blocking=True)

        input = th.cat([inp[k] for k in in_keys], dim=-1)
        if d.prior:
            fwd_args = dict(
                y=lbl if model.y_cond else None, fp16=False, decode=False
            )
            with th.no_grad():
                out, mloss, metrics = model(input, **fwd_args)
            iv = None
            outs = [out]
            extra = {}
        else:
            with th.no_grad():
                out, extra = model(input)
            outs = [out]
            mloss = None
            metrics = {}

        loss, lmetrics = compute_loss(
            d, inp, mask, outs, mloss, extra.get('kl', None)
        )
        metrics.update(**lmetrics)

        n = mask.sum().item()
        eval_nsmpl += n
        eval_mloss.append(loss * n)

        # Render videos with reconstructed motions
        if eval_batch < 2 and d.cfg.eval.video:
            for i in range(out.shape[1]):
                # TODO make relative positions absolute again...
                if in_keys == ['qpos']:
                    env.set_pose(
                        out[0, i].cpu().numpy(), input[0, i].cpu().numpy()
                    )
                else:
                    outd = gym.spaces.unflatten(
                        d.obs_space, out[0, i].cpu().numpy()
                    )
                    inpd = gym.spaces.unflatten(
                        d.obs_space, input[0, i].cpu().numpy()
                    )
                    env.set_mocap_pose(outd, inpd)
                d.rq.push(
                    th.from_numpy(env.render().copy()),
                    s_left=[f'Updates {updates}', 'Eval'],
                    s_right=[f'Batch {eval_batch}', f'Frame {i}'],
                )

        for k, v in metrics.items():
            if v.numel() > 0:
                eval_metrics[k].append(v.mean().cpu())
            else:
                eval_metrics[k].append(v.cpu())

        eval_batch += 1
    d.rq.plot()

    metrics = {k: th.stack(v).mean() for k, v in eval_metrics.items()}
    for k, v in metrics.items():
        d.tbw.add_scalar(f'Eval/{k}', v, updates)
    mloss = sum(l.sum().item() for l in eval_mloss) / eval_nsmpl

    if not d.prior:
        rlosses = []
        for i in range(100):
            try:
                rlosses.append(metrics[f'recons_loss_l{i+1}'].item())
            except:
                break
        rloss = '/'.join([f'{r:.03f}' for r in rlosses])
        if d.cfg.loss == 'comic':
            crs = []
            for i in range(100):
                try:
                    crs.append(metrics[f'comic_reward_l{i+1}'].mean().item())
                except:
                    break
            cr = '/'.join([f'{r:.03f}' for r in crs])
            log.info(f'Eval {updates} cr {cr} rloss {rloss}')
        else:
            log.info(f'Eval {updates} rloss {rloss}')
        return
    if not model_acc.x_cond:
        log.info(f'Eval {updates} loss {mloss:.03f}')
        return

    raise NotImplementedError()


def train_loop(d):
    device = d.cfg.device
    model = d.model
    optim = d.optim._all_
    sched = d.sched._all_
    updates = 0
    # assert len(d.cfg.dataset.inputs) == 1
    in_keys = d.cfg.dataset.inputs
    cp_path = d.cfg.checkpoint_path
    model_acc = model
    if dist.is_initialized():
        model_acc = model.module

    try:
        l_bins = model_acc.bottleneck.level_blocks[0].k_bins
    except:
        l_bins = 0
    epoch = 0

    d.bms = hm.load_amass_bms('/checkpoint/jgehring/data/smplh')
    mj_betas = np.load('/checkpoint/jgehring/data/smplh/mjshape.npz')
    NUM_BETAS = 16
    skel_betas = (
        th.from_numpy(mj_betas['male']).to(th.float32).view(1, NUM_BETAS)
    )
    d.skel = amass.create_skeleton_from_amass_bodymodel(
        d.bms['male'], skel_betas, len(amass.joint_names), amass.joint_names
    )
    d.xform_from_parent_joint = {
        j.name: th.from_numpy(j.xform_from_parent_joint).float().cuda()
        for j in d.skel.joints
    }
    d.bms['male'].to(device)
    # d.bms['female'].to(device)

    # Feature normalization
    if d.normalize:
        means = d.dl['train'].dataset.means
        stds = d.dl['train'].dataset.stds
        if d.prior:
            vae = model_acc.vae
        else:
            vae = model_acc
        if (vae.input_mean == 0).all():
            vae.input_mean.copy_(
                th.cat(
                    [th.from_numpy(means[k]).to(device) for k in in_keys],
                    dim=-1,
                )
            )
            vae.input_std.copy_(
                th.cat(
                    [th.from_numpy(stds[k]).to(device) for k in in_keys], dim=-1
                )
            )

    # Load from checkpoint?
    if Path(cp_path).is_file():
        with open(cp_path, 'rb') as f:
            data = th.load(
                f, map_location={'cuda:0': f'cuda:{th.cuda.current_device()}'}
            )
        model_acc.load_state_dict(data['model'])
        optim.load_state_dict(data['optim'])
        sched.load_state_dict(data['sched'])
        updates = data.get('updates', 0)
        epoch = data.get('epoch', 0)
        log.info(f'Continue training from checkpoint, {updates}')

    env = ReplayEnv()
    fwd_loss = d.cfg.loss
    if (
        fwd_loss.startswith('comic')
        or fwd_loss.startswith('l2_')
        or fwd_loss in ('hmvae', 'quat', 'smpl')
    ):
        fwd_loss = 'l2'

    def barrier():
        if dist.is_initialized():
            dist.barrier()

    def checkpoint():
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        with open(f'{cp_path}.tmp', 'wb') as fd:
            th.save(
                {
                    'model': model_acc.state_dict(),
                    'optim': optim.state_dict(),
                    'sched': sched.state_dict(),
                    'updates': updates,
                    'epoch': epoch,
                },
                fd,
            )
        os.rename(f'{cp_path}.tmp', cp_path)
        p = Path(cp_path)
        cp_unique_path = str(p.with_name(p.stem + f'_{updates:07d}' + p.suffix))
        shutil.copy(cp_path, cp_unique_path)

    def cp_eval():
        checkpoint()
        barrier()
        model.eval()
        eval(d, updates, env)
        model.train()
        barrier()

    # Initial checkpoint and eval
    cp_eval()

    # Training loop
    if d.ds['train'] is not None:
        d.ds['train'].set_epoch(epoch)
    it = iter(d.dl['train'])
    did_just_eval = True
    rendered = False
    zocc = None
    while updates <= d.cfg.n_updates:
        at_end = False
        try:
            inp, lbl, mask = next(it)
        except StopIteration:
            at_end = True

        # Evaluate
        if not did_just_eval and (
            (d.eval_interval > 0 and updates % d.eval_interval == 0)
            or (d.eval_interval <= 0 and at_end)
        ):
            cp_eval()
            rendered = False
            zocc = None
            did_just_eval = True
        else:
            did_just_eval = False

        if at_end:
            epoch += 1
            if d.ds['train'] is not None:
                d.ds['train'].set_epoch(epoch)
            it = iter(d.dl['train'])
            continue

        # Training setup
        lbl = lbl.to(device, non_blocking=True)
        inp = {k: v.to(device, non_blocking=True) for k, v in inp.items()}
        inp = {k: v.squeeze() for k, v in inp.items()}
        if 'rpos' in inp:
            xy_base = inp['rpos'].narrow(1, 0, 1).narrow(2, 0, 2).clone()
            inp['rpos'].narrow(2, 0, 2).sub_(xy_base)
        if 'relxypos' in inp:
            inp['relxypos'].narrow(2, 0, 2)
        mask = mask.to(device, non_blocking=True)

        input = th.cat([inp[k] for k in in_keys], dim=-1)
        if d.prior:
            fwd_args = dict(
                y=lbl if model.y_cond else None, fp16=False, decode=False
            )
            out, mloss, metrics = model(input, **fwd_args)
            iv = None
            outs = [None]
            extra = {}
        else:
            out, extra = model(input)
            outs = [out]
            mloss = None
            metrics = {'entropy': extra['entropy']}

        loss, lmetrics = compute_loss(
            d, inp, mask, outs, mloss, extra.get('kl', None)
        )
        metrics.update(**lmetrics)

        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()

        if not rendered and d.cfg.video:
            rendered = True
            for i in range(out.shape[1]):
                if in_keys == ['qpos']:
                    env.set_pose(
                        out.detach()[0, i].cpu().numpy(),
                        input[0, i].cpu().numpy(),
                    )
                else:
                    outd = gym.spaces.unflatten(
                        d.obs_space, out.detach()[0, i].cpu().numpy()
                    )
                    inpd = gym.spaces.unflatten(
                        d.obs_space, input[0, i].cpu().numpy()
                    )
                    env.set_mocap_pose(outd, inpd)
                d.rq.push(
                    th.from_numpy(env.render().copy()),
                    s_left=[f'Epoch {epoch}', 'Train'],
                    s_right=[f'Batch {eval_batch}', f'Frame {i}'],
                )
            d.rq.plot()

        if updates % 100 == 0 and d.prior:
            loss = metrics['loss'].item()
            log.info(f'Update {updates}/{epoch} loss {loss:.03f}')
        elif updates % 100 == 0 and not d.prior:
            loss = metrics['loss'].item()
            with th.no_grad():
                umask = mask.unsqueeze(-1)
                minp = input.masked_select(umask)
                moutp = out.masked_select(umask)
                l1 = metrics.get('l1_loss', F.l1_loss(moutp, minp)).item()
                l2 = metrics.get('l2_loss', F.mse_loss(moutp, minp)).item()
            entropy = metrics['entropy'].item()
            if d.cfg.loss == 'comic':
                crs = []
                for i in range(100):
                    try:
                        crs.append(
                            metrics[f'comic_reward_l{i+1}'].mean().item()
                        )
                    except:
                        break
                cr = '/'.join([f'{r:.03f}' for r in crs])
                log.info(
                    f'Update {updates}/{epoch} cr {cr} l1/l2 {l1:.03f}/{l2:.03f} entropy {entropy:.03f}'
                )
            else:
                log.info(
                    f'Update {updates}/{epoch} loss {loss:.03f} l1/l2 {l1:.03f}/{l2:.03f} entropy {entropy:.03f}'
                )

            for k, v in metrics.items():
                if v.numel() > 0:
                    d.tbw.add_scalar(f'Train/{k}', v.mean(), updates)
                else:
                    d.tbw.add_scalar(f'Train/{k}', v, updates)
        updates += 1


def setup_training(cfg: DictConfig):
    viz = Visdom(
        server=f'http://{cfg.visdom.host}',
        port=cfg.visdom.port,
        env=cfg.visdom.env,
        offline=cfg.visdom.offline,
        log_to_filename=cfg.visdom.logfile,
    )
    log.info('Starting render queue')
    rq = hucc.RenderQueue(viz)

    log.info('Constructing datasets')
    data_cls = {
        'frame': OfflineMocapFrameDataset,
        'stack': OfflineMocapStackDataset,
        'seq': OfflineMocapSequenceDataset,
        'subseq': OfflineMocapSubSequenceDataset,
    }[cfg.dataset.type]
    obs_space = None
    action_space = None
    data = {}
    if cfg.get('validation_data', True):
        subsets = ['train', 'valid']
    else:
        subsets = ['']

    for subset in subsets:
        log.info(f'Constructing dataset {subset} with {data_cls}')
        data[subset] = data_cls(
            path=cfg.dataset.path,
            inputs=cfg.dataset.inputs,
            label=cfg.dataset.label,
            prefix=subset,
            **cfg.dataset.get(f'{subset}_args', cfg.dataset.get('args')),
        )
        if obs_space is None:
            obs_space = data[subset].observation_space
            action_space = data[subset].action_space
    log.info('Constructing model')
    model = hucc.make_model(cfg.model, obs_space, action_space)
    log.info(f'Model from config:\n{model}')
    model.to(cfg.device)
    optim, sched = hucc.make_optim(cfg.optim, model)

    if cfg.init_from:
        with open(cfg.init_from, 'rb') as fd:
            d = th.load(fd)
            model.load_state_dict(d['model'])

    if dist.is_initialized():
        ds = {
            k: DistributedSampler(
                data[k],
                seed=cfg.seed,
                shuffle=True if k in ('train', '') else False,
            )
            for k in data.keys()
        }
    else:
        ds = {k: None for k in data.keys()}
    if data_cls.collate is not None:
        collate_fn = {
            k: lambda items: data_cls.collate(
                items,
                cfg.dataset.get(f'{k}_args', cfg.dataset.get('args')).length,
            )
            for k in data.keys()
        }
    else:
        collate_fn = {k: None for k in data.keys()}
    dl = {
        k: DataLoader(
            data[k],
            num_workers=cfg.dataset.num_workers,
            batch_size=cfg.batch_size,
            shuffle=True if k in ('train', '') and ds[k] is None else False,
            sampler=ds[k],
            collate_fn=collate_fn[k],
            pin_memory=True,
            drop_last=True if k in ('train', '') else False,
        )
        for k in data.keys()
    }

    tbw = SummaryWriter(cfg.get('tb_dir', 'tb'))

    return SimpleNamespace(
        model=model,
        optim=optim,
        sched=sched,
        dl=dl,
        ds=ds,
        cfg=cfg,
        tbw=tbw,
        viz=viz,
        rq=rq,
        obs_space=obs_space,
        action_space=action_space,
        **cfg.loop_args,
    )


def worker_(
    cfg: DictConfig, rank: int = 0, size: int = 1, rdvu_path: str = None
):
    device_id = 0
    log.info(f'Starting worker {rank+1}/{size}')
    if size > 1:
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=size,
            init_method=f'file://{rdvu_path}',
        )
        device_id = rank % th.cuda.device_count()
        th.cuda.set_device(rank % th.cuda.device_count())

    th.manual_seed(cfg.seed)
    ld = setup_training(cfg)

    if size > 1:
        ld.model = th.nn.parallel.DistributedDataParallel(
            ld.model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=cfg.loop_args.prior,
        )
        with ld.model.join():
            train_loop(ld)
        dist.destroy_process_group()
        os.kill(os.getpid(), 9)  # Fix hanging workers
    else:
        train_loop(ld)


def worker(
    cfg: DictConfig, rank: int = 0, size: int = 1, rdvu_path: str = None
):
    try:
        worker_(cfg, rank, size, rdvu_path)
    except:
        log.exception(f'!!! Unhandled exception in worker {rank}/{size}:')
        raise


def device_count(x):
    return th.cuda.device_count()


@hydra.main(config_path='config')  # , version_base='1.1')
def main(cfg: DictConfig):
    log.info(f'** running from source tree at {hydra.utils.get_original_cwd()}')
    log.info(f'** configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    jobid = os.environ.get('SLURM_JOBID', uuid.uuid4())
    rscount = os.environ.get('SLURM_RESTART_COUNT', 0)
    rdvu_file = f'{cfg.distributed.rdvu_path}/rdvu-{jobid}-{rscount}'

    n_procs = 1
    if cfg.distributed.size == 'auto':
        with mp.Pool(1) as p:
            n_procs = p.map(device_count, [None])[0]
        OmegaConf.set_struct(cfg, False)
        cfg.batch_size //= n_procs
        OmegaConf.set_struct(cfg, True)
    else:
        n_procs = cfg.distributed.size

    if n_procs == 1:
        worker(cfg)
        return

    procs = []
    for i in range(n_procs):
        procs.append(
            mp.Process(target=worker, args=(cfg, i, n_procs, rdvu_file))
        )
        procs[-1].start()
    for proc in procs:
        proc.join()
        log.info('proc done')
    try:
        os.remove(rdvu_file)
    except:
        pass


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error(
        'Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback)
    )

    # Enter post-portem debugger unless we're running on slurm
    if os.environ.get('SLURM_JOBID', None):
        return

    import pdb

    pdb.post_mortem(exc_traceback)


if __name__ == '__main__':
    import sys

    sys.excepthook = handle_exception
    main()
