# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from copy import copy, deepcopy
from os import path as osp
from typing import Any, Dict

import h5py
import hdf5plugin
import numpy as np
import torch as th
import tqdm
from fairmotion.ops.motion import resample
from omegaconf import DictConfig, OmegaConf

import hucc.mocap
from hucc.envs.amass import datasets
from hucc.mocap.datasets import merge_labels
from train_zprior import setup_training


def featurize_clip(args):
    clip, bms, mj_betas, dt, rotate = args
    motion = hucc.mocap.amass_to_motion(clip, bms, mj_betas, 1 / dt)
    if rotate != 0:
        motion = hucc.mocap.rotate_motion_z(motion, rotate)
    try:
        comic_feats = hucc.mocap.motion_to_comic(motion)
    except ValueError as e:
        print(clip, e)
        return None, None, None, None
    jpos = hucc.mocap.motion_to_jposrot(motion)
    return clip, comic_feats, jpos


def collate_labels(
    ann: Dict[str, Any],
    lmap: Dict[str, int],
    level: str,
    key: str,
    num_steps: int,
    dt: float,
    max_labels: int,
):
    if level == 'frame_seq':
        level = 'frame'
        if ann is None or ann.get('frame_ann', None) is None:
            level = 'seq'
    akey = f'{level}_ann'
    if ann is None or ann.get(akey, None) is None:
        return np.zeros((num_steps, max_labels), dtype=np.int32)
    labels = ann[akey]['labels']
    lseqs = np.zeros((len(labels), num_steps), dtype=np.int32)
    for i, l in enumerate(labels):
        if l is None:
            continue
        if key == 'act_cat_pri':
            if l.get('act_cat', None) is None:
                continue
            id = lmap[l['act_cat'][0]]
        elif key == 'act_cat':
            if l.get('act_cat', None) is None:
                continue
            id = lmap['/'.join(l['act_cat'])]
        else:
            if l.get(key, None) is None:
                continue
            id = lmap[l[key]]
        if level == 'seq':
            lseqs[i] = id
        else:
            start = max(0, int(l['start_t'] / dt + 0.5))
            end = min(int(l['end_t'] / dt + 0.5), num_steps)
            lseqs[i, start:end] = id
    merged = merge_labels(lseqs, max_labels)
    if merged.shape[0] < max_labels:
        merged = np.concatenate(
            [
                merged,
                np.zeros(
                    (max_labels - merged.shape[0], num_steps),
                    dtype=merged.dtype,
                ),
            ],
            axis=0,
        )
    return merged.T


def encode_clip(setup, ofeats, mode: str):
    cfg = setup.cfg
    in_keys = cfg.dataset.inputs
    skip = cfg.dataset.train_args.skip
    win = cfg.dataset.train_args.length
    feats = copy(ofeats)
    zs_per_step = skip / (2 ** cfg.model.downs_t[0])
    if zs_per_step != float(int(zs_per_step)):
        raise ValueError('skip and downs_t don\'t match up')
    zs_per_step = int(zs_per_step)
    zs_half = int(np.floor(0.5 * win / (2 ** cfg.model.downs_t[0])))
    downs_t = (
        cfg.model.repr.downs_t
        if hasattr(cfg.model, 'repr')
        else cfg.model.downs_t
    )

    if mode == 'relxypos':
        inp = th.cat([th.from_numpy(feats[k]) for k in in_keys], dim=-1)
        pad = 2 ** sum(downs_t)
        npad = pad - (inp.shape[0] % pad)
        if npad > 0:
            inp = th.cat([inp, th.zeros(npad, inp.shape[1])], dim=0)
        with th.no_grad():
            _, _, ims, _ = setup.model(inp.cuda().unsqueeze(0), loss_fn='l2')
        ret = {}
        for level in range(setup.model.levels):
            xs = ims['xs'][level][0].T.cpu().numpy()
            xqs = ims['xs_quantised'][level][0].T.cpu().numpy()
            zs = ims['zs'][level].cpu().numpy()
            nsub = 2 ** sum(downs_t[: level + 1])
            if nsub > 1:
                xs = np.repeat(xs, nsub, axis=0)
                xqs = np.repeat(xqs, nsub, axis=0)
                zs = np.repeat(zs, nsub, axis=0)
            ntgt = feats['rpos'].shape[0]
            xs = xs[:ntgt]
            xqs = xqs[:ntgt]
            zs = zs[:ntgt]
            ret[f'xs{level}'] = xs
            ret[f'xqs{level}'] = xqs
            ret[f'zs{level}'] = zs

        for k in list(ret.keys()):
            ret[f'{k}_tanh'] = np.tanh(ret[k])
        return ret

    inputs = []
    offsets = []
    # TODO sane decoding of *all* inputs?
    for i in range(0, feats['rpos'].shape[0] - win, skip):
        if mode == 'rposrelxy':
            offsets.append(feats['rpos'][0, 0:2].copy())
            feats['rpos'] = feats['rpos'].copy()
            feats['rpos'][:, 0:2] -= offsets[-1]
        elif mode == 'abspos' or mode == 'abs' or mode == 'relxypos':
            pass
        else:
            raise ValueError(f'Unknown encoding mode: {mode}')
        inputs.append(
            th.cat(
                [th.from_numpy(feats[k][i : i + win]) for k in in_keys], dim=-1
            )
        )
    if len(inputs) == 0:
        raise ValueError(
            f'Empty inputs list for clip of size {feats["rpos"].shape[0]}'
        )
    input = th.stack(inputs, dim=0).cuda()
    with th.no_grad():
        tout, _, iv, _ = setup.model(input, loss_fn='l2')
    tz = iv['zs']
    if tz[0].dtype != th.long:
        tz = [i.permute(0, 2, 1) for i in iv['zs']]
    tx = [i.permute(0, 2, 1) for i in iv['xs']]
    txq = [i.permute(0, 2, 1) for i in iv['xs_quantised']]

    outs = []
    zs = []
    xs = []
    xqs = []
    for i in range(tout.shape[0]):
        out, z, x, xq = tout[i], tz[0][i], tx[0][i], txq[0][i]
        if mode == 'rposrelxy':
            out[:, 0:2] += th.from_numpy(offsets[i]).to(out.device)
        if len(outs) == 0:
            outs.append(out[: win // 2])
            zs.append(z[:zs_half])
            xs.append(x[:zs_half])
            xqs.append(xq[:zs_half])
        else:
            outs.append(out[win // 2 : win // 2 + skip])
            zs.append(z[zs_half : zs_half + zs_per_step])
            xs.append(x[zs_half : zs_half + zs_per_step])
            xqs.append(xq[zs_half : zs_half + zs_per_step])
    outs.append(out[win // 2 + skip :])
    zs.append(z[zs_half + zs_per_step :])
    xs.append(x[zs_half + zs_per_step :])
    xqs.append(xq[zs_half + zs_per_step :])
    return [th.cat(v, 0).cpu().numpy() for v in [outs, zs, xs, xqs]]


def main():
    parser = argparse.ArgumentParser(
        description='Convert AMASS clips to MuJoCo reference traces'
    )
    parser.add_argument('clips', type=str, nargs='*', help='AMASS clips')
    parser.add_argument(
        '--smplh-path',
        type=str,
        default=osp.join(os.getcwd(), 'hucc/envs/assets/smplh'),
    )
    parser.add_argument('--output-path', type=str, default='output.h5')
    parser.add_argument(
        '--mjshape-path',
        type=str,
        default=osp.join(os.getcwd(), 'hucc/envs/assets/robot-smplh-shape.npz'),
    )
    parser.add_argument('--dt', type=float, default=0.03)
    parser.add_argument('-j', type=int, default=mp.cpu_count())
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--mode', type=str, default='relxypos')
    parser.add_argument('--rotate', type=float, default=0)
    parser.add_argument('--compress', default=False, action='store_true')
    parser.add_argument('-f', '--clips-file', type=str, default='')
    parser.add_argument('--subset', type=str, default='')

    args = parser.parse_args()
    bms = hucc.mocap.load_amass_bms(args.smplh_path)
    mj_betas = {
        'male': np.load(args.mjshape_path)['male'],
        'female': np.load(args.mjshape_path)['female'],
    }

    ds_kwargs = {}
    if args.compress:
        ds_kwargs = hdf5plugin.Zstd()

    # Load mjbox model from checkpoint
    if args.checkpoint:
        cfg = OmegaConf.load(
            os.path.dirname(args.checkpoint) + '/.hydra/config.yaml'
        )
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {
                    'dataset': {
                        'args': {'in_memory': False},
                        'train_args': {'in_memory': False},
                        'valid_args': {'in_memory': False},
                    },
                    'visdom': {'offline': False},
                }
            ),
        )
        setup = setup_training(cfg)
        with open(args.checkpoint, 'rb') as fd:
            d = th.load(fd)
            missing, unexpected = setup.model.load_state_dict(
                d['model'], strict=False
            )
            if missing == ['input_mean', 'input_std']:
                print('Copy input stats from checkpoint')
                setup.model.input_mean.copy_(d['data_mean'])
                setup.model.input_std.copy_(d['data_std'])
            else:
                assert len(missing) == 0
        setup.model.eval()
    else:
        setup = None

    if args.clips_file:
        if args.clips_file == '-':
            with sys.stdin as f:
                clips = [l.strip() for l in f.readlines()]
        else:
            with open(args.clips_file, 'r') as f:
                clips = [l.strip() for l in f.readlines()]
    else:
        clips = args.clips

    if args.subset:
        subset = set(datasets[args.subset])
        clips_subset = []
        for clip in clips:
            corpus, cname = hucc.mocap.corpus_clip_name(clip)
            if f'{corpus}_{cname}' in subset:
                clips_subset.append(clip)
        clips = clips_subset

    pool = mp.Pool(args.j)
    with h5py.File(args.output_path, 'w') as outf:
        seq = [(clip, bms, mj_betas, args.dt, args.rotate) for clip in clips]
        for res in pool.imap_unordered(featurize_clip, seq):
            clip, feats, jfeats = res
            if clip is None:
                continue
            corpus, cname = hucc.mocap.corpus_clip_name(clip)
            if setup is not None:
                try:
                    encs = encode_clip(setup, jfeats, args.mode)
                except Exception as e:
                    print(e)
                    continue
            else:
                encs = {}

            print(f'Storing clip {corpus}_{cname}')

            # We'll need to create some empty groups to satisfy the dm_control
            # data loading logic.
            grp = outf.create_group(f'{corpus}_{cname}')
            grp.attrs['day'] = 0
            grp.attrs['month'] = 0
            grp.attrs['year'] = 0
            grp.attrs['dt'] = args.dt
            grp.attrs['mjbox'] = args.checkpoint
            num_steps = feats['joints'].shape[0]
            grp.attrs['num_steps'] = num_steps
            grp.create_group('props')
            grp = grp.create_group('walkers')
            grp = grp.create_group('walker_0')
            grp.create_group('markers')
            grp.create_group('scaling').create_group('subtree_0')

            for k, v in feats.items():
                if k == 'qpos':
                    continue
                v = v.T
                grp.create_dataset(
                    k, shape=v.shape, dtype=v.dtype, data=v, **ds_kwargs
                )
            for k, v in jfeats.items():
                v = v.T
                grp.create_dataset(
                    f'ref_{k}',
                    shape=v.shape,
                    dtype=v.dtype,
                    data=v,
                    **ds_kwargs,
                )
            for k, v in encs.items():
                v = v.T
                grp.create_dataset(
                    f'mjbox_{k}',
                    shape=v.shape,
                    dtype=v.dtype,
                    data=v,
                    **ds_kwargs,
                )


if __name__ == '__main__':
    main()
