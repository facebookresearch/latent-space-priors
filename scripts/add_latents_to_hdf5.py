# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import shutil
from os import path as osp

import h5py
import hdf5plugin
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from train_zprior import setup_training


def load_from_checkpoint(path, **config):
    cfg = OmegaConf.load(osp.dirname(path) + '/.hydra/config.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.create(config))
    setup = setup_training(cfg)
    with open(path, 'rb') as fd:
        d = th.load(fd)
        try:
            missing, unexpected = setup.model.load_state_dict(
                d['model'], strict=False
            )
            # print(missing, unexpected)
            assert unexpected == []
            assert (
                missing == []
                or missing == ['vae.input_mean', 'vae.input_std']
                or missing == ['input_mean', 'input_std']
            )
            if d.get('data_mean', None) is not None:
                setup.data_mean = d['data_mean'].cpu()
                setup.data_std = d['data_std'].cpu()
        except:
            m = nn.Module()
            m.repr = nn.Sequential(nn.Identity(), setup.model.repr)
            m.load_state_dict(d['model'])
    return setup


def encode_clip_relxypos(setup, feats, level=0):
    cfg = setup.cfg
    in_keys = cfg.dataset.inputs
    downs_t = [0]
    inp = th.cat([th.from_numpy(feats[k]) for k in in_keys], dim=-1)
    pad = 2 ** sum(downs_t)
    npad = pad - (inp.shape[0] % pad)
    if npad > 0:
        inp = th.cat([inp, th.zeros(npad, inp.shape[1])], dim=0)
    with th.no_grad():
        if hasattr(cfg.model, 'vae'):
            _, extra = setup.model.vae(inp.cuda().unsqueeze(0))
        else:
            _, extra = setup.model(inp.cuda().unsqueeze(0))
    zd = extra['zdist'][0].T.cpu().numpy()
    z = extra['z'][0].T.cpu().numpy()
    ntgt = next(iter(feats.values())).shape[0]
    zd = zd[:ntgt]
    z = z[:ntgt]
    return {'z': z, 'z_dist': zd}


def encode_training_data(setup, dataset_path):
    outs = {}
    in_keys = setup.cfg.dataset.inputs
    with h5py.File(dataset_path) as f:
        if 'start' in f.keys():
            subsets = ['']
        else:
            subsets = [f'{k}/' for k in f.keys()]
        for subset in tqdm(subsets):
            start = np.concatenate(
                [f[f'{subset}/start'][:], [f[f'{subset}/rpos'].shape[0]]]
            )
            feats = {}
            for k in in_keys:
                feats[k] = f[f'{subset}/{k}'][:]

            xqs = []
            zs = []
            for s, e in tqdm(
                zip(start[:-1], start[1:]),
                total=len(start) - 1,
                desc='Encoding data',
                leave=False,
            ):
                enc = encode_clip_relxypos(
                    setup, {k: v[s:e] for k, v in feats.items()}
                )
                xqs.append(enc['z'])
                zs.append(enc['z_dist'])
            outs[subset] = {
                'xqs': np.concatenate(xqs),
                'zs': np.concatenate(zs),
                #'feats': (start, relxypos, r6)
            }
    return outs


def main():
    parser = argparse.ArgumentParser(
        description='Convert AMASS clips to MuJoCo reference traces'
    )
    parser.add_argument('--checkpoint', default='checkpoint.pt')
    parser.add_argument('input')
    parser.add_argument('output')

    args = parser.parse_args()

    setup = load_from_checkpoint(
        args.checkpoint,
        **{
            'dataset': {
                'args': {'in_memory': False},
                'train_args': {'in_memory': False},
                'valid_args': {'in_memory': False},
            },
            'visdom': {'offline': False},
        },
    )

    outs = encode_training_data(setup, args.input)

    shutil.copy(args.input, args.output)
    with h5py.File(args.output, 'r+') as f:
        for subset, data in tqdm(outs.items()):
            grp = f[subset]
            grp.attrs['mjbox'] = args.checkpoint
            for k, v in data.items():
                grp.create_dataset(k, shape=v.shape, dtype=v.dtype, data=v)


if __name__ == '__main__':
    main()
