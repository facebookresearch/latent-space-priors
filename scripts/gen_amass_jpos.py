# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import multiprocessing as mp
import os
import sys
from os import path as osp

import h5py
import hdf5plugin
import numpy as np
import torch as th
import tqdm
from fairmotion.data import amass, bvh

import hucc.mocap
from hucc.envs.amass import datasets


def amass_to_jpos(clip_path, bms, mj_betas, target_fps):
    if clip_path.endswith('.bvh'):
        NUM_BETAS = 16
        skel_betas = (
            th.from_numpy(mj_betas['male']).to(th.float32).view(1, NUM_BETAS)
        )
        skel = amass.create_skeleton_from_amass_bodymodel(
            bms['male'], skel_betas, len(amass.joint_names), amass.joint_names
        )
        motion = bvh.load(clip_path)
        motion.set_skeleton(skel)
    else:
        motion = hucc.mocap.amass_to_motion(
            clip_path, bms, mj_betas, target_fps
        )
    return hucc.mocap.motion_to_jposrot(motion)


def call_amass_to_jpos(args):
    try:
        data = amass_to_jpos(*args)
    except:
        logging.exception(f'Error processing clip: {args}')
        return None
    data['clip_id'] = clip_path_to_id(args[0])
    return data


def clip_path_to_id(clip_path):
    if clip_path.endswith('.bvh'):
        return (
            f'{osp.basename(osp.dirname(clip_path))}_{osp.basename(clip_path)}'
        )
    corpus = osp.basename(osp.dirname(osp.dirname(clip_path)))
    cname = osp.basename(clip_path).removesuffix('_poses.npz')
    if corpus == 'CMU':
        subject = osp.basename(osp.dirname(clip_path))
        # Make sure we're compatible with comic's naming
        cnames = cname.split('_')
        # These are not in comic but have equal clip names
        if subject == '18_19_Justin':
            cnames[0] = '18'
        elif subject == '18_19_rory':
            cnames[0] = '19'
        elif subject == '20_21_Justin1':
            cnames[0] = '20'
        elif subject == '20_21_rory1':
            cnames[0] = '21'
        elif subject == '22_23_justin':
            cnames[0] = '22'
        elif subject == '22_23_Rory':
            cnames[0] = '23'
        cname = f'{int(cnames[0]):03d}_{cnames[1]}'
    return f'{corpus}_{cname}'


def main():
    parser = argparse.ArgumentParser(
        description='Convert AMASS clips to MuJoCo reference traces'
    )
    parser.add_argument('clips', type=str, nargs='*', help='AMASS clips')
    parser.add_argument(
        '--smplh-path', type=str, default='hucc/envs/assets/smplh'
    )
    parser.add_argument('--output-path', type=str, default='jpos.h5')
    parser.add_argument(
        '--mjshape',
        type=str,
        default=osp.join(os.getcwd(), 'hucc/envs/assets/robot-smplh-shape.npz'),
    )
    parser.add_argument('--dt', type=float, default=0.03)
    parser.add_argument('-j', type=int, default=mp.cpu_count())
    parser.add_argument('--compress', default=False, action='store_true')
    parser.add_argument('-f', '--clips-file', type=str, default='')
    parser.add_argument('--subset', type=str, default='')

    args = parser.parse_args()

    bms = hucc.mocap.load_amass_bms(args.smplh_path)
    mjshape = args.mjshape
    if mjshape in {'zero', 'zeroes', 'zeros'}:
        mj_betas = {
            'male': np.zeros((16,), dtype=np.float32),
            'female': np.zeros((16,), dtype=np.float32),
        }
    elif mjshape == 'clip':
        mj_betas = 'clip'
    else:
        mj_betas = {
            'male': np.load(mjshape)['male'],
            'female': np.load(mjshape)['female'],
        }

    ds_kwargs = {}
    if args.compress:
        ds_kwargs = hdf5plugin.Zstd()

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
    create_group = True
    clip_ids = []
    with h5py.File(args.output_path, 'w') as outf:
        outf.create_dataset('start', shape=(len(clips),), dtype='l')
        seq = [(clip, bms, mj_betas, 1 / args.dt) for clip in clips]
        n = 0
        p = 0
        pbar = tqdm.tqdm(total=len(seq))
        for data in pool.imap_unordered(call_amass_to_jpos, seq):
            # for s in seq:
            # data = call_amass_to_jpos(s)
            if data is None:
                continue
            if create_group:
                for k, v in data.items():
                    if k == 'clip_id':
                        continue
                    outf.create_dataset(
                        k,
                        shape=(1, v.shape[-1]),
                        maxshape=(None, v.shape[-1]),
                        chunks=True,
                        dtype=np.float32,
                        **ds_kwargs,
                    )
                create_group = False
            l = data['jpos'].shape[0]
            outf['start'][n] = p
            p += l
            n += 1
            for k, v in data.items():
                if k == 'clip_id':
                    continue
                outf[k].resize(p, 0)
                outf[k][-l:] = v.astype(np.float32)
            clip_ids.append(data['clip_id'])
            pbar.update()

        outf.create_dataset(
            'clip_id', data=[id.encode('ascii') for id in clip_ids]
        )


if __name__ == '__main__':
    logging.basicConfig()
    main()
