# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import gzip
import json
import logging
import os
import shutil
import sys
from bisect import bisect_right
from collections import defaultdict
from functools import partial
from os import path as osp
from os.path import dirname
from typing import Dict

import bisk
import gym
import h5py
import hdf5plugin
import numpy as np
import torch as th
from bisk.helpers import add_robot, root_with_floor
from dm_control.locomotion import arenas
from dm_control.locomotion.mocap.cmu_mocap_data import get_path_for_cmu
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose.cmu_subsets import \
    CMU_SUBSETS_DICT
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.rl.control import PhysicsError
from fairmotion.data import amass
from fairmotion.ops import conversions
from human_body_prior.body_model.body_model import BodyModel
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import rotation_conversions as rc
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

import hucc
from hucc.mocap.amass import corpus_clip_name
from hucc.mocap.conversion import JOINT_MAP
from hucc.spaces import box_space, th_flatten, th_unflatten

try:
    import bcolz

    _has_bcolz = True
except ModuleNotFoundError:
    _has_bcolz = False
    pass

try:
    sys.path.insert(0, dirname(dirname(dirname(__file__))) + '/cpp/build')
    import hucc_cpp_ext as hucc_cpp
except ModuleNotFoundError:
    hucc_cpp = None


log = logging.getLogger(__name__)


class ReplayEnv(bisk.BiskEnv):
    def __init__(self):
        super().__init__()

        smplh_path = (
            '/private/home/jgehring/projects/mocap/hucc/hucc/envs/assets/smplh'
        )
        self.bm = {
            'male': BodyModel(
                bm_fname=osp.join(smplh_path, 'male/model.npz'), num_betas=16
            ).to('cpu'),
            'female': BodyModel(
                bm_fname=osp.join(smplh_path, 'female/model.npz'), num_betas=16
            ).to('cpu'),
        }
        mjshape_path = '/private/home/jgehring/projects/mocap/hucc/mjshape.npz'
        self.mj_betas = np.load(mjshape_path)
        self.skel = {
            'male': amass.create_skeleton_from_amass_bodymodel(
                self.bm['male'],
                th.tensor(self.mj_betas['male']).view(1, -1).to(th.float32),
                len(amass.joint_names),
                amass.joint_names,
            ),
            'female': amass.create_skeleton_from_amass_bodymodel(
                self.bm['female'],
                th.tensor(self.mj_betas['female']).view(1, -1).to(th.float32),
                len(amass.joint_names),
                amass.joint_names,
            ),
        }

        root = root_with_floor()
        _, robot_pos = add_robot(root, 'HumanoidAMASSPC', 'robot')
        _, ghost_pos = add_robot(root, 'HumanoidAMASSPC', 'ghost')
        root.asset.material['ghost/self'].rgba = [1, 1, 1, 0.5]
        root.find('default', 'ghost/humanoid').geom.set_attributes(
            conaffinity=0, contype=0
        )

        frameskip = 5
        fs = root.find('numeric', 'robot/frameskip')
        if fs is not None:
            frameskip = int(fs.data[0])
        # with open('/tmp/out.xml', 'wt') as f:
        #    f.write(root.to_xml_string())
        self.init_sim(root, frameskip)

        self.init_qpos[0:3] = [0.0, 0.0, 1.3]
        self.init_qpos[3:7] = [0.859, 1.0, 1.0, 0.859]
        self.init_qpos[63 : 63 + 3] = [0.0, 2.0, 1.3]
        self.init_qpos[63 + 3 : 63 + 7] = [0.859, 1.0, 1.0, 0.859]

    def set_pose(self, qpos, qpos_ref):
        self.p.data.qpos[:63] = qpos
        self.p.data.qpos[63:] = qpos_ref

        try:
            self.p.forward()
        except PhysicsError as e:
            log.exception(e)

    def set_mocap_pose(self, pose, pose_ref):
        qpos = hucc.mocap.jposrot_to_qpos(pose, self.p)[:63]
        qpos_ref = hucc.mocap.jposrot_to_qpos(pose_ref, self.p)[:63]
        self.p.data.qpos[:63] = qpos
        self.p.data.qpos[63:] = qpos_ref

        '''
        prefix = {0: 'robot', 63: 'ghost'}
        g = 'male'
        for p, d in zip([pose, pose_ref], [0, 63]):
            x = p['fwd'].reshape(-1, 3)
            # Ensure x and z are orthogonal
            z = p['up'].reshape(-1, 3) - x * (
                np.einsum('bx,bx->b', x, p['up'].reshape(-1, 3))
                / np.einsum('bx,bx->b', x, x)
            ).reshape(-1, 1)
            y = np.cross(z, x)
            R = np.stack([x, y, z], axis=-1)

            if 'jpos' in p:
                self.p.data.qpos[d + 0 : d + 3] = p['jpos'][0:3]
            else:
                self.p.data.qpos[d + 0 : d + 3] = p['rpos']
            self.p.data.qpos[d + 3 : d + 7] = conversions.R2Q(R[0])[
                [3, 0, 1, 2]
            ]
            for mj_joint, src in JOINT_MAP.items():
                src_joint, src_idx, src_order = src
                import ipdb
                ipdb.set_trace()
                r = conversions.R2E(R[self.skel[g].get_index_joint(src_joint)])
                self.p.named.data.qpos[f'{prefix[d]}/{mj_joint}'] = r[src_idx]
        '''
        try:
            self.p.forward()
        except PhysicsError as e:
            log.exception(e)


def featurize_reference(d):
    from dm_control.utils.transformations import quat_to_axisangle

    logging.info('Loading reference data')
    ref_path = get_path_for_cmu(version='2020')
    walker_type = cmu_humanoid.CMUHumanoidPositionControlledV2020
    arena = arenas.Floor()
    task = tracking.MultiClipMocapTracking(
        walker=walker_type,
        arena=arena,
        ref_path=ref_path,
        dataset='all',
        ref_steps=[0],
        min_steps=10,
        reward_type='comic',
        ghost_offset=None,
        always_init_at_clip_start=True,
    )

    refs = h5py.File(get_path_for_cmu(version='2020'), 'r')
    clips = CMU_SUBSETS_DICT['all'].ids
    body_idxs = task._.body_idxs

    # Extract absolute features for all clips
    import ipdb

    ipdb.set_trace()
    ref_features = {}
    for clip in clips:
        ref = refs[clip]['walkers']['walker_0']
        n_frames = ref['joints'].shape[1]
        reference_rel_joints = ref['joints'][:][
            task._walker.mocap_to_observable_joint_order
        ].T
        rel_bodies_pos_global = ref['body_positions'][:][body_idxs].T
        time_in_clip = np.linspace(0, 1, n_frames)
        rel_root_pos_local = ref['position'][:].T

    # Assemble concatenation for rollout data
    ref_keys = [
        'reference_rel_joints',
        'rel_bodies_pos_global',
        'time_in_clip',
        'rel_root_pos_local',
        'rel_root_quat',
        'rel_bodies_quats',
        'appendages_pos',
        'rel_bodies_pos_local',
        'ego_bodies_quats',
    ]
    self.d['reference'] = None
    N = self.d['clip_id'].shape[0]

    for start, end in zip(d['start'], list(d['start'][1:]) + [self.N]):
        for t in range(end - start):
            # XXX Make relative things... relative?

            p += 1


class OfflineMocapDataset(Dataset):
    def __init__(self, path, **kwargs):
        self.f = h5py.File(path, 'r')
        prefix = kwargs.get('prefix', None)

        datasets = []

        def find_ds(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        if prefix:
            self.f[prefix].visititems(find_ds)
            self.d = {k: self.f[prefix][k] for k in datasets}
        else:
            self.f.visititems(find_ds)
            self.d = {k: self.f[k] for k in datasets}
        if kwargs.get('load_reference', False):
            self.d['reference'] = featurize_reference(self.d)

        self._means, self._stds = {}, {}
        items = self.f[prefix].items() if prefix else self.f.items()
        for k, v in items:
            try:
                self._means[k] = v.attrs['mean'][:]
                self._stds[k] = v.attrs['std'][:]
            except:
                pass

    @property
    def means(self):
        return self._means

    @property
    def stds(self):
        return self._stds


class OfflineMocapFrameDataset(OfflineMocapDataset):
    collate = None

    def __init__(
        self,
        path,
        inputs,
        label: str,
        in_memory: bool = True,
        comic_ref_path=None,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.inputs = inputs
        if isinstance(self.inputs, DictConfig):
            self.inputs = dict(self.inputs)
        self.label = label
        self.N = self.d[label].shape[0]
        if comic_ref_path:
            self.comic_d = h5py.File(comic_ref_path, 'r')
        else:
            self.comid_d = None
        self.starts = self.d['start'][:]
        self.clips = [s.decode('utf8') for s in self.d['clip_id'][:]]
        self._comic_feature: Dict[str, np.ndarray] = {}

        if isinstance(self.inputs, dict):
            self.features = []
            for k, v in self.inputs.items():
                self.features += v
        else:
            self.features = list(self.inputs)

        feature_shapes = {}
        for f in self.features:
            if f.startswith('comic:/'):
                feature_shapes[f] = self.comic_feature(
                    f[7:], self.clips[0]
                ).shape
            else:
                feature_shapes[f] = (self.d[f].shape[-1],)

        spaces = []
        if isinstance(self.inputs, dict):
            for k, v in self.inputs.items():
                shapes = [feature_shapes[f] for f in v]
                ndim = len(shapes[0])
                assert all(len(s) == ndim for s in shapes), shapes
                cat_dim = sum(s[ndim - 1] for s in shapes)
                if ndim > 1:
                    assert all(
                        s[0:-1] == shapes[0][0:-1] for s in shapes
                    ), shapes
                    spaces.append((k, box_space(shapes[0][0:-1] + (cat_dim,))))
                else:
                    spaces.append((k, box_space((cat_dim,))))
        else:
            for inp in self.inputs:
                spaces.append((inp, box_space(feature_shapes[inp])))

        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = box_space((self.d[label].shape[-1],))

        # Load data into memory for faster access
        self.frames = None
        for k in self.features + [label]:
            if k.startswith('comic:/'):
                self.frames = self.d['observation/frame'][:]
                break
        if in_memory:
            for k in self.features + [label]:
                if k.startswith('comic:/'):
                    continue
                if isinstance(self.d[k], h5py._hl.dataset.Dataset):
                    if _has_bcolz:
                        self.d[k] = bcolz.carray(self.d[k])
                    else:
                        self.d[k] = self.d[k][:]

    def apply_comic_args(self, f, idx, *args):
        sidx, eidx = idx, idx + 1
        flatten = True
        for arg in args:
            k, v = arg.split('=')
            if k == 'shift':
                t = int(v)
                sidx += int(t)
                eidx += int(t)
            elif k == 'stack':
                eidx = sidx + int(v)
            elif k == 'flatten':
                flatten = v.lower() in ('yes', 'true', 't', '1')
            else:
                raise NotImplementedError(f'Comic feature argument {k}')
        if sidx >= f.shape[0]:
            f = np.repeat(f[-1], eidx - sidx).reshape(-1, f.shape[1])
        elif eidx >= f.shape[0]:
            a = f[sidx:]
            b = np.repeat(f[-1], eidx - sidx - a.shape[0]).reshape(
                -1, f.shape[1]
            )
            f = np.concatenate([a, b], axis=0)
        else:
            f = f[sidx:eidx]
        if flatten:
            return f.flatten()
        return f

    def comic_feature(self, name, clip_id, idx=0):
        key, *args = name.split(',')
        p = key.replace('$clip', clip_id)
        cached = self._comic_feature.get(p, None)
        if cached is not None:
            return self.apply_comic_args(cached, idx, *args)
        assert self.comic_d is not None
        f = self.comic_d[p][:].T
        self._comic_feature[p] = f
        return self.apply_comic_args(f, idx, *args)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        finp = {}
        for k in self.features:
            if k.startswith('comic:/'):
                clip = self.clips[bisect_right(self.starts, idx) - 1]
                finp[k] = self.comic_feature(k[7:], clip, self.frames[idx])
            else:
                finp[k] = th.tensor(self.d[k][idx])
        inp = {}
        if isinstance(self.inputs, dict):
            for k, v in self.inputs.items():
                inp[k] = np.concatenate([finp[f] for f in v], axis=-1)
        else:
            inp = finp
        datum = (inp, th.tensor(self.d[self.label][idx]), th.tensor([True]))
        return datum


class OfflineMocapStackDataset(OfflineMocapDataset):
    collate = None

    def __init__(self, path, inputs, label, k: int, **kwargs):
        super().__init__(path, **kwargs)
        self.inputs = inputs
        self.label = label
        self.k = k
        self.N = self.d[label].shape[0]
        self.pad = {}
        for start in self.d['start'][1:]:
            for i in range(k - 1):
                self.pad[start - i - 1] = k - i - 1
        for i in range(k - 1):
            self.pad[self.N - i - 1] = k - i - 1

        self.observation_space = gym.spaces.Dict(
            [(inp, box_space((k, self.d[inp].shape[-1]))) for inp in inputs]
        )
        self.action_space = box_space((self.d[label].shape[-1],))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        didx = list(range(self.k))
        npad = self.pad.get(idx, 0)
        if npad > 0:
            didx[-npad:] = [didx[-npad - 1]] * npad
        inp = {}
        for k in self.inputs:
            d = self.d[k][idx : min(idx + self.k, self.N)]
            inp[k] = d[didx]
        return (inp, th.tensor(self.d[self.label][idx]), th.tensor([True]))


def merge_labels(labels: np.ndarray, max_labels: int) -> np.ndarray:
    merged = [labels[0]]
    for l in labels[1:]:
        # If no overlap with a current sequence, merge; otherwise, if less than
        # max_sequences exist, create a new sequence; else, replace items at
        # sequence with minimal overlap.
        overlaps = [((m != 0) & (l != 0)).sum() for m in merged]
        mino = min(overlaps)
        idx = np.argmin(overlaps)
        if mino == 0:
            merged[idx] += l
        elif len(merged) < max_labels:
            merged.append(l)
        else:
            merged[idx][np.where(l != 0)] = l[np.where(l != 0)]
    return np.stack(merged)


class OfflineMocapSequenceDataset(OfflineMocapDataset):
    def __init__(self, path, inputs, label, max_labels=1, **kwargs):
        super().__init__(path, **kwargs)
        self.inputs = inputs
        self.label = label
        self.starts = list(self.d['start'][:])
        self.ends = self.starts[1:]
        self.ends.append(self.d[inputs[0]].shape[0])

        self.action_space = box_space((self.d[label].shape[-1],))
        self.observation_space = gym.spaces.Dict(
            [(inp, box_space((self.d[inp].shape[-1],))) for inp in inputs]
        )

    def __len__(self):
        return len(self.starts) - 1

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = self.ends[idx]
        return {
            k: th.tensor(self.d[k][start:end]) for k in self.inputs
        }, th.tensor(self.d[self.label][start:end])

    @staticmethod
    def collate(items, minlength=-1):
        B = len(items)
        items = sorted(items, key=lambda item: -item[1].shape[0])
        lens = [item[1].shape[0] for item in items]
        T = max(lens[0], minlength)

        inp = {}
        if items[0][1].ndim == 1:
            lbl = th.zeros((B, T), dtype=items[0][1].dtype)
        else:
            lbl = th.zeros(B, T, items[0][1].shape[-1], dtype=items[0][1].dtype)
        mask = th.zeros(B, T, dtype=th.bool)
        for i, item in enumerate(items):
            if i == 0:
                for k, v in item[0].items():
                    inp[k] = th.zeros((B, T) + v.shape[1:])
                    inp[k][0, 0 : v.shape[0]] = v
            else:
                for k, v in item[0].items():
                    inp[k][i, 0 : v.shape[0]] = v
            lbl[i, 0 : item[1].shape[0]] = item[1]
            mask[i, 0 : item[1].shape[0]].fill_(True)

        return inp, lbl, mask


class OfflineMocapSubSequenceDataset(OfflineMocapSequenceDataset):
    def __init__(
        self,
        path,
        inputs,
        label,
        length: int,
        skip: int = 1,
        stack: int = 1,
        in_memory: bool = True,
        **kwargs,
    ):
        super().__init__(path, inputs, label, **kwargs)

        self.k = stack
        self.N = self.d[inputs[0]].shape[0]
        self.pad = defaultdict(dict)
        for inp in inputs:
            for start in self.d['start'][1:]:
                for i in range(self.k - 1):
                    self.pad[inp][start - i - 1] = self.k - i - 1
            for i in range(self.k - 1):
                self.pad[inp][self.N - i - 1] = self.k - i - 1

        self.newstarts = []
        self.newends = []
        for start, end in zip(self.starts, self.ends):
            for i in range(start, end, skip):
                self.newstarts.append(i)
                self.newends.append(min(i + length, end))

        self.starts = self.newstarts
        self.ends = self.newends

        # Load data into memory for faster access
        if in_memory:
            for k in inputs + [label]:
                if isinstance(self.d[k], h5py._hl.dataset.Dataset):
                    if _has_bcolz:
                        self.d[k] = bcolz.carray(self.d[k])
                    else:
                        self.d[k] = self.d[k][:]

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = self.ends[idx]
        didx = []
        for i in range(self.k):
            didx.append(list(range(end - start)))
            npad = self.pad.get(end - self.k + i, 0)
            if npad > 0:
                import ipdb

                ipdb.set_trace()
                # TODO not this way
                didx[-1][-npad:] = [
                    didx[-1][max(-npad - 1, -len(didx[-1]))]
                ] * min(npad, len(didx[-1]))

        inps = []
        for i in range(self.k):
            inps.append(
                {
                    x: th.tensor(
                        self.d[x][start + i : min(end + i, self.N)][didx[i]]
                    )
                    for x in self.inputs
                }
            )

        out = (
            {x: th.stack([inp[x] for inp in inps], dim=1) for x in self.inputs},
            th.tensor(self.d[self.label][start:end]),
        )
        return out


def th_FUp2R(fwd, up, rpos):
    x = fwd.view(-1, 3)
    z = up.view(-1, 3) - x * (
        th.einsum('bx,bx->b', x, up.view(-1, 3)) / th.einsum('bx,bx->b', x, x)
    ).view(-1, 1)
    y = th.cross(z, x)
    zeros = th.zeros_like(x[:, 0]).unsqueeze(1)
    x = th.cat([x, zeros], dim=1)
    y = th.cat([y, zeros], dim=1)
    z = th.cat([z, zeros], dim=1)
    pcol = th.zeros_like(z)
    pcol[:, 3].add_(1)
    pcol.view(-1, 22, 4)[:, 0, 0:3].add_(rpos)
    R = th.stack([x, y, z, pcol], dim=-1)
    if fwd.shape[-1] == 3:
        R = R.view((fwd.shape[:-1]) + (4, 4))
    return R


def th_R62T(A):
    a1 = th.atleast_2d(A.narrow(-1, 0, 3))
    a2 = th.atleast_2d(A.narrow(-1, 3, 3))
    c1 = a1 / th.linalg.norm(a1, axis=-1, keepdims=True)
    c2 = a2 - (th.einsum('...bx,...bx->...b', c1, a2).unsqueeze(-1) * c1)
    c2 = c2 / th.linalg.norm(c2, axis=-1, keepdims=True)
    c3 = th.cross(c1, c2)
    c4 = th.zeros_like(c1)
    zrow = th.zeros(c4.shape[:-1] + (1, 4), dtype=c4.dtype, device=c4.device)
    zrow[..., -1].add_(1)
    R = th.cat([th.stack([c1, c2, c3, c4], dim=-1), zrow], dim=-2)
    if A.ndim == 1:
        R = R[0]
    return R


def th_R62R(A):
    a1 = th.atleast_2d(A.narrow(-1, 0, 3))
    a2 = th.atleast_2d(A.narrow(-1, 3, 3))
    c1 = a1 / th.linalg.norm(a1, axis=-1, keepdims=True)
    c2 = a2 - (th.einsum('...bx,...bx->...b', c1, a2).unsqueeze(-1) * c1)
    c2 = c2 / th.linalg.norm(c2, axis=-1, keepdims=True)
    c3 = th.cross(c1, c2)
    R = th.stack([c1, c2, c3], dim=-1)
    if A.ndim == 1:
        R = R[0]
    return R


def th_R2R6(R):
    return th.cat([R[..., 0], R[..., 1]], axis=-1)


def th_T2Q(T):
    R = T.narrow(-2, 0, 3).narrow(-1, 0, 3)
    return rc.matrix_to_quaternion(R)


def th_Q2T(Q):
    ldims = Q.shape[:-1]
    R = rc.quaternion_to_matrix(Q.view(-1, 4)).view(ldims + (3, 3))
    T = th.zeros(ldims + (4, 4), device=Q.device)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1
    return T


def th_FUp2T(fwd, up, rpos, skel, xform_from_parent_joint):
    Tlocal = th_FUp2R(fwd.view(-1, 22, 3), up.view(-1, 22, 3), rpos.view(-1, 3))

    # From fairmotion.Pose.get_transform(local=False)
    T = []
    for i, joint in enumerate(skel.joints):
        Tj = th.matmul(xform_from_parent_joint[joint.name], Tlocal[:, i])
        while joint.parent_joint is not None:
            Tj_t = th.matmul(
                xform_from_parent_joint[joint.parent_joint.name],
                Tlocal[:, skel.get_index_joint(joint.parent_joint)],
            )
            Tj = th.matmul(Tj_t, Tj)
            joint = joint.parent_joint
        T.append(Tj)
    return th.stack(T, dim=1)


def th_Qp2T(quat, rpos, skel, xform_from_parent_joint):
    Tlocal = th_Q2T(quat.view(-1, 22, 4))
    Tlocal[:, :, :3, 3].add_(rpos.unsqueeze(-2))

    # From fairmotion.Pose.get_transform(local=False)
    T = []
    for i, joint in enumerate(skel.joints):
        Tj = th.matmul(xform_from_parent_joint[joint.name], Tlocal[:, i])
        while joint.parent_joint is not None:
            Tj_t = th.matmul(
                xform_from_parent_joint[joint.parent_joint.name],
                Tlocal[:, skel.get_index_joint(joint.parent_joint)],
            )
            Tj = th.matmul(Tj_t, Tj)
            joint = joint.parent_joint
        T.append(Tj)
    return th.stack(T, dim=1)


def th_R6p2T(r6, rpos, skel, xform_from_parent_joint):
    Tlocal = th_R62T(r6.view(-1, 22, 6))
    Tlocal[:, :, :3, 3].add_(rpos.unsqueeze(-2))

    # From fairmotion.Pose.get_transform(local=False)
    T = []
    for i, joint in enumerate(skel.joints):
        Tj = th.matmul(xform_from_parent_joint[joint.name], Tlocal[:, i])
        while joint.parent_joint is not None:
            Tj_t = th.matmul(
                xform_from_parent_joint[joint.parent_joint.name],
                Tlocal[:, skel.get_index_joint(joint.parent_joint)],
            )
            Tj = th.matmul(Tj_t, Tj)
            joint = joint.parent_joint
        T.append(Tj)
    return th.stack(T, dim=1)


def th_R62T_global(r6, skel, xform_from_parent_joint):
    Tlocal = th_R62T(r6.view(-1, 22, 6))

    # From fairmotion.Pose.get_transform(local=False)
    T = []
    for i, joint in enumerate(skel.joints):
        Tj = th.matmul(xform_from_parent_joint[joint.name], Tlocal[:, i])
        while joint.parent_joint is not None:
            Tj_t = th.matmul(
                xform_from_parent_joint[joint.parent_joint.name],
                Tlocal[:, skel.get_index_joint(joint.parent_joint)],
            )
            Tj = th.matmul(Tj_t, Tj)
            joint = joint.parent_joint
        T.append(Tj)
    return th.stack(T, dim=1)


def hmvae_loss(orig, recon, skel, xform_from_parent_joint, joints_factor=10):
    assert 'rpos' in orig and 'rpos' in recon
    assert 'relxypos' in orig and 'relxypos' in recon
    assert 'r6' in orig and 'r6' in recon

    # Forward kinematics
    if 'r6' in orig:
        To = th_R62T_global(orig['r6'], skel, xform_from_parent_joint)
        Tr = th_R62T_global(recon['r6'], skel, xform_from_parent_joint)

    To = To[: Tr.shape[0]]
    N = Tr.shape[0]

    loss_r6 = (orig['r6'] - recon['r6']).square().view(N, -1)
    # TODO: relative z as well?
    loss_vel = (
        (orig['relxypos'][:, :2] - recon['relxypos'][:, :2])
        .square()
        .view(N, -1)
    )
    loss_pos = (orig['rpos'] - recon['rpos']).square().view(N, -1)

    loss_rots = (To[:, :, :3, :3] - Tr[:, :, :3, :3]).square().view(N, -1)
    loss_joints = (To[:, :, :3, 3] - Tr[:, :, :3, 3]).square().view(N, -1)

    return {
        'loss_hmvae': loss_r6.mean()
        + loss_vel.mean()
        + loss_pos.mean()
        + loss_rots.mean()
        + joints_factor * loss_joints.mean(),
        'loss_r6': loss_r6,
        'loss_vel': loss_vel,
        'loss_pos': loss_pos,
        'loss_rots': loss_rots,
        'loss_joints': loss_joints,
    }


def quat_loss(orig, recon, skel, xform_from_parent_joint):
    if hucc_cpp is None:
        raise RuntimeError('C++ extensions could not be loaded')

    To = th_Qp2T(orig['quat'], orig['rpos'], skel, xform_from_parent_joint)
    Tr = th_Qp2T(recon['quat'], recon['rpos'], skel, xform_from_parent_joint)
    To = To[: Tr.shape[0]]
    N = Tr.shape[0]

    err_quats = (
        hucc_cpp.bounded_quat_dist2(
            orig['quat'].view(-1, 4), recon['quat'].view(-1, 4)
        )
        .square()
        .mean(dim=-1)
    )
    err_bodies = (
        (To[:, :, :3, 3] - Tr[:, :, :3, 3]).view(N, -1).square().mean(dim=1)
    )

    return {
        'loss': err_quats.mean() + err_bodies.mean(),
        'loss_quats': err_quats,
        'loss_bodies': err_bodies,
    }


def smpl_loss(orig, recon, bmodel):
    if hucc_cpp is None:
        raise RuntimeError('C++ extensions could not be loaded')

    if 'r6' in orig:
        A_orig = rc.matrix_to_axis_angle(th_R62R(orig['r6']))
        A_recon = rc.matrix_to_axis_angle(th_R62R(recon['r6']))
    elif 'quat' in orig:
        A_orig = rc.quaternion_to_axis_angle(orig['quat'])
        A_recon = rc.quaternion_to_axis_angle(orig['quat'])
    else:
        raise NotImplementedError()
    N = A_orig.shape[0]

    v_orig = bmodel(
        trans=orig['rpos'],
        root_orient=A_orig[:, 0],
        pose_body=A_orig[:, 1:].reshape(N, -1),
    ).v
    v_recon = bmodel(
        trans=orig['rpos'],
        root_orient=A_recon[:, 0],
        pose_body=A_recon[:, 1:].reshape(N, -1),
    ).v
    loss_v = (v_orig - v_recon).square().view(N, -1).mean(dim=-1)

    if 'r6' in orig:
        loss_rot = (orig['r6'] - recon['r6']).square().view(N, -1).mean(dim=-1)
    elif 'quat' in orig:
        loss_rot = (
            hucc_cpp.bounded_quat_dist2(
                orig['quat'].view(-1, 4), recon['quat'].view(-1, 4)
            )
            .square()
            .mean(dim=-1)
        )

    return {
        'loss': loss_v.mean() + loss_rot.mean(),
        'loss_v': loss_v,
        'loss_rot': loss_rot,
    }
    return 0


def comic_reward(orig, recon, skel, xform_from_parent_joint):
    if hucc_cpp is None:
        raise RuntimeError('C++ extensions could not be loaded')

    if 'r6' in orig:
        To = th_R6p2T(orig['r6'], orig['rpos'], skel, xform_from_parent_joint)
        Tr = th_R6p2T(recon['r6'], recon['rpos'], skel, xform_from_parent_joint)
    elif 'quat' in orig:
        To = th_Qp2T(orig['quat'], orig['rpos'], skel, xform_from_parent_joint)
        Tr = th_Qp2T(
            recon['quat'], recon['rpos'], skel, xform_from_parent_joint
        )
    else:
        To = th_FUp2T(
            orig['fwd'], orig['up'], orig['rpos'], skel, xform_from_parent_joint
        )
        Tr = th_FUp2T(
            recon['fwd'],
            recon['up'],
            recon['rpos'],
            skel,
            xform_from_parent_joint,
        )
    To = To[: Tr.shape[0]]
    N = Tr.shape[0]

    # Approximate Comic features
    termination_threshold = 0.3
    err_bodies = (
        (To[:, :, :3, 3] - Tr[:, :, :3, 3]).abs().view(N, -1).mean(dim=1)
    )
    # Well, the joint err should be just the difference in qpos... error
    # of rotation matrix isn't very nice
    # err_joints = (To[:,:,:3,:3] - Tr[:,:,:3,:3]).abs().view(N, -1).mean(dim=1)
    if 'r6' in orig:
        err_joints = (orig['r6'] - recon['r6']).abs().view(N, -1).mean(dim=1)
    elif 'quat' in orig:
        # Let's just skip it in this case
        err_joints = err_bodies
    else:
        err_joints = (orig['fwd'] - recon['fwd']).abs().view(N, -1).mean(
            dim=1
        ) + (orig['up'] - recon['up']).abs().view(N, -1).mean(dim=1)
    termination_error = 0.5 * err_bodies + 0.5 * err_joints
    term_reward = 1.0 - termination_error / termination_threshold

    com_o = To[:, 0, :3, 3]  # just the root position
    com_r = Tr[:, 0, :3, 3]
    diff_com = (com_o - com_r).square()
    diff_com2 = (orig['rpos'] - recon['rpos']).square().view(N, -1)
    # print(diff_com)

    diff_bodies = (To[:, 1:, :3, 3] - Tr[:, 1:, :3, 3]).square().view(N, -1)
    if 'r6' in orig:
        diff_joints = (orig['r6'] - recon['r6']).square().view(N, -1)
    else:
        diff_joints = th.zeros((N, 1), device=orig['rpos'].device)
    diff_jointsmat = (To[:, :, :3, :3] - Tr[:, :, :3, :3]).square().view(N, -1)

    joint_idx = {j.name: i for i, j in enumerate(skel.joints)}
    apps = ['ltoe', 'rtoe', 'lwrist', 'rwrist', 'upperneck']
    appendages_o = th.stack([To[:, joint_idx[j], :3, 3] for j in apps], dim=1)
    appendages_r = th.stack([Tr[:, joint_idx[j], :3, 3] for j in apps], dim=1)
    diff_app = (appendages_o - appendages_r).square().view(N, -1)
    # print(diff_app)

    body_quats_o = th_T2Q(To[:, 1:])
    body_quats_r = th_T2Q(Tr[:, 1:])
    bq = hucc_cpp.bounded_quat_dist2(
        body_quats_o.view(-1, 4), body_quats_r.view(-1, 4)
    ).square()
    diff_bq = bq.view(body_quats_o.shape[:-1])
    # print(diff_bq)

    diff_vels = []
    for key in ('avel', 'lvel', 'rvel'):
        if key in orig:
            diff_vels.append((orig[key] - recon[key]).square().view(N, -1))
    if not diff_vels:
        diff_vel = th.zeros_like(diff_bq)
    else:
        diff_vel = th.cat(diff_vels, -1)

    cr = 0.5 * term_reward + 0.5 * (
        0.1 * th.exp(-10 * diff_com.sum(dim=1))
        + 0.15 * th.exp(-40 * diff_app.sum(dim=1))
        + 0.65 * th.exp(-2 * diff_bq.sum(dim=1))
        + 1.0 * th.exp(-0.1 * diff_vel.sum(dim=1))
    )
    return {
        'comic_reward': cr,
        'term_reward': term_reward,
        'diff_com': diff_com,
        'diff_com2': diff_com2,
        'diff_app': diff_app,
        'diff_bq': diff_bq,
        'diff_bodies': diff_bodies,
        'diff_joints': diff_joints,
        'diff_jointsmat': diff_jointsmat,
        'diff_vel': diff_vel,
    }
