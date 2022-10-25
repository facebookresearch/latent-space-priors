# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
from collections import defaultdict
from os import path as osp

import h5py
import hdf5plugin
import numpy as np


def merge_h5(path, files):
    data = defaultdict(list)
    start_off = 0
    for f in files:
        fpath = osp.join(path, f) + '.h5'
        with h5py.File(fpath, 'r') as inf:
            got_len = False
            t_start_off = start_off
            for k, v in inf.items():
                if k == 'start':
                    v = v[:] + t_start_off
                elif not got_len:
                    start_off += v.shape[0]
                    got_len = True
                data[k].append(v[:])
    return {k: np.concatenate(v, axis=0) for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser('Merge H5 sequence corpora')
    parser.add_argument('--output-path', default='output.h5')
    parser.add_argument('--path', default=os.getcwd())
    parser.add_argument(
        '--train',
        default='CMU,MPI_Limits,TotalCapture,Eyes_Japan_Dataset,KIT,BMLmovi,BMLhandball,EKUT',
    )
    parser.add_argument('--valid', default='Transitions_mocap,SSM_synced')
    parser.add_argument('--test', default='HumanEva,MPI_HDM05,SFU,MPI_mosh')

    args = parser.parse_args()

    with h5py.File(args.output_path, 'w') as outf:
        grp = outf.create_group('train')
        data = merge_h5(args.path, args.train.split(','))
        for k, v in data.items():
            if k == 'clip_id':
                grp.create_dataset(k, data=v)
            else:
                grp.create_dataset(
                    k, shape=v.shape, dtype=v.dtype, data=v, **hdf5plugin.Zstd()
                )

        grp = outf.create_group('valid')
        data = merge_h5(args.path, args.valid.split(','))
        for k, v in data.items():
            if k == 'clip_id':
                grp.create_dataset(k, data=v)
            else:
                grp.create_dataset(
                    k, shape=v.shape, dtype=v.dtype, data=v, **hdf5plugin.Zstd()
                )

        if args.test:
            grp = outf.create_group('test')
            data = merge_h5(args.path, args.test.split(','))
            for k, v in data.items():
                if k == 'clip_id':
                    grp.create_dataset(k, data=v)
                else:
                    grp.create_dataset(
                        k,
                        shape=v.shape,
                        dtype=v.dtype,
                        data=v,
                        **hdf5plugin.Zstd()
                    )


if __name__ == '__main__':
    main()
