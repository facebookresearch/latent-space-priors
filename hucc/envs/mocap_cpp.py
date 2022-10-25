# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import sys
from collections import defaultdict
from copy import copy, deepcopy
from os.path import dirname
from typing import Any, Dict, List, Tuple

import bisk.helpers
import dm_env
import gym
import h5py
import hdf5plugin
import numpy as np
import torch as th
from dm_control import composer, mjcf
from dm_control.locomotion import arenas
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import cmu_subsets, tracking
from dm_control.locomotion.tasks.reference_pose.types import ClipCollection
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.walkers.initializers import UprightInitializer
from gym.utils import seeding

from hucc.envs import amass
from hucc.spaces import box_space

sys.path.insert(0, dirname(dirname(dirname(__file__))) + '/cpp/build')
import hucc_cpp_ext as hucc_cpp

log = logging.getLogger(__name__)


_WALKER_GEOM_GROUP = 2
_WALKER_INVIS_GROUP = 1
_CMU_MOCAP_JOINTS = (
    'lfemurrz',
    'lfemurry',
    'lfemurrx',
    'ltibiarx',
    'lfootrz',
    'lfootrx',
    'ltoesrx',
    'rfemurrz',
    'rfemurry',
    'rfemurrx',
    'rtibiarx',
    'rfootrz',
    'rfootrx',
    'rtoesrx',
    'lowerbackrz',
    'lowerbackry',
    'lowerbackrx',
    'upperbackrz',
    'upperbackry',
    'upperbackrx',
    'thoraxrz',
    'thoraxry',
    'thoraxrx',
    'lowerneckrz',
    'lowerneckry',
    'lowerneckrx',
    'upperneckrz',
    'upperneckry',
    'upperneckrx',
    'headrz',
    'headry',
    'headrx',
    'lclaviclerz',
    'lclaviclery',
    'lhumerusrz',
    'lhumerusry',
    'lhumerusrx',
    'lradiusrx',
    'lwristry',
    'lhandrz',
    'lhandrx',
    'lfingersrx',
    'lthumbrz',
    'lthumbrx',
    'rclaviclerz',
    'rclaviclery',
    'rhumerusrz',
    'rhumerusry',
    'rhumerusrx',
    'rradiusrx',
    'rwristry',
    'rhandrz',
    'rhandrx',
    'rfingersrx',
    'rthumbrz',
    'rthumbrx',
)


class AMASSHumanoid(cmu_humanoid.CMUHumanoidPositionControlledV2020):
    """A CMU humanoid walker."""

    def _build(
        self,
        name='walker',
        marker_rgba=None,
        include_face=False,
        initializer=None,
    ):
        self._mjcf_root = mjcf.from_path(self._xml_path)
        if name:
            self._mjcf_root.model = name
        # DMC will add a freejoint for this robot automatically, and we can only
        # have one.
        self._mjcf_root.worldbody.body[0].freejoint.remove()

        if marker_rgba is not None:
            for geom in self.marker_geoms:
                geom.set_attributes(rgba=marker_rgba)

        self._actuator_order = np.argsort(_CMU_MOCAP_JOINTS)
        self._inverse_order = np.argsort(self._actuator_order)

        try:
            self._initializers = tuple(initializer)
        except TypeError:
            self._initializers = (initializer or UprightInitializer(),)

        if include_face:
            head = self._mjcf_root.find('body', 'head')
            head.add(
                'geom',
                type='capsule',
                name='face',
                size=(0.065, 0.014),
                pos=(0.000341465, 0.048184, 0.01),
                quat=(0.717887, 0.696142, -0.00493334, 0),
                mass=0.0,
                contype=0,
                conaffinity=0,
            )

            face_forwardness = head.pos[1] - 0.02
            head_geom = self._mjcf_root.find('geom', 'head')
            nose_size = head_geom.size[0] / 4.75
            face = head.add(
                'body', name='face', pos=(0.0, 0.039, face_forwardness)
            )
            face.add(
                'geom',
                type='capsule',
                name='nose',
                size=(nose_size, 0.01),
                pos=(0.0, 0.0, 0.0),
                quat=(1, 0.7, 0, 0),
                mass=0.0,
                contype=0,
                conaffinity=0,
                group=_WALKER_INVIS_GROUP,
            )

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find('body', 'torso')

    @composer.cached_property
    def mocap_tracking_bodies(self):
        # remove root body
        root_body = self._mjcf_root.find('body', 'torso')
        return tuple(
            b for b in self._mjcf_root.find_all('body') if b != root_body
        )

    @property
    def _xml_path(self):
        return bisk.helpers.asset_path() + '/humanoidamasspc.xml'
        # return _XML_PATH.format(model_version='2019')


_OBS_PROPRIO = [
    'appendages_pos',
    'body_height',
    'joints_pos',
    'joints_vel',
    'gyro_control',
    'joints_vel_control',
    'velocimeter_control',
    'sensors_touch',
    'sensors_velocimeter',
    'sensors_gyro',
    'sensors_accelerometer',
    'end_effectors_pos',
    'actuator_activation',
    'sensors_torque',
    'world_zaxis',
]
_OBS_COMIC_REF = [
    'reference_rel_joints',
    'reference_rel_bodies_pos_global',
    'reference_rel_root_pos_local',
    'reference_rel_root_quat',
    'reference_rel_bodies_quats',
    'reference_appendages_pos',
    'reference_rel_bodies_pos_local',
    'reference_ego_bodies_quats',
]


class CMUMocap2020CppMulti(gym.Env):
    def __init__(
        self,
        dataset: str = 'comic_walk_tiny',
        n_ref: int = 5,
        ref_offset: int = 1,
        ref_lag: int = 1,
        past_ref: bool = False,
        time_limit: float = 30,
        features: Dict[str, List[str]] = None,
        min_steps: int = 10,
        reward_type: str = 'comic',
        ghost: bool = False,
        always_init_at_clip_start: bool = False,
        clips: str = None,
        reference_dims: int = 1,
        ref_path: str = None,
        robot: str = 'V2020',
        n_envs: int = 1,
        device: str = 'cuda',
        verbose: bool = True,
        last_ref_only: bool = False,
        sample_by: str = None,
        init_noise: float = 0,
        init_offset: Tuple[int, int] = None,
        init_state: str = 'mocap',
        early_termination: bool = True,
        end_with_mocap: bool = True,
    ):
        if reference_dims not in (1, 2):
            raise ValueError(
                f'reference_dims should be 1 or 2, is ${reference_dims}'
            )
        if last_ref_only:
            assert reference_dims == 2
        self.last_ref_only = last_ref_only

        if robot.lower() == 'v2020':
            walker_type = cmu_humanoid.CMUHumanoidPositionControlledV2020
        elif robot.lower() == 'amass':
            walker_type = AMASSHumanoid
        else:
            raise ValueError(f'Unknown robot type {robot}')
        arena = arenas.Floor()

        # Fetch available clips
        if not ref_path:
            ref_path = cmu_mocap_data.get_path_for_cmu(version='2020')
        with h5py.File(ref_path, 'r') as f:
            all_clips = set(f.keys())
        asset_path = dirname(__file__) + '/assets'
        with open(f'{asset_path}/amass.json') as f:
            metadata = json.load(f)

        filter = None
        if ':' in dataset:
            dataset, filter_spec = dataset.split(':')
            filter = filter_spec.split('=')

        if clips is not None:
            dataset = ClipCollection(ids=clips.split(':'))
        elif dataset.startswith('comic_'):
            dataset = deepcopy(cmu_subsets.CMU_SUBSETS_DICT[dataset[6:]])
        elif dataset == 'all':
            dataset = ClipCollection(ids=list(all_clips))
        elif dataset in amass.datasets:
            dataset = ClipCollection(ids=amass.datasets[dataset])
        else:
            dlow = dataset.lower()
            clips = [c for c in all_clips if c.lower().startswith(dlow)]
            if (not clips) and (
                dataset in ['walk_tiny', 'locomotion_small', 'run_jump_tiny']
            ):
                # Support old configs
                dataset = deepcopy(cmu_subsets.CMU_SUBSETS_DICT[dataset])
            else:
                dataset = ClipCollection(ids=clips)

        remove = set()

        # Apply meta-data filter
        if filter:
            for id in dataset.ids:
                md = metadata.get(id, {})
                if md.get(filter[0], '') != filter[1]:
                    remove.add(id)

        # Filter missing clips that may have been specified in a dataset
        n_clips = len(dataset.ids)
        for clip in dataset.ids:
            if clip not in all_clips:
                log.info(f'Clip {clip} not in reference, discarding')
                remove.add(clip)
        if len(remove) > 0:
            log.info(
                f'Discarded {len(remove)} of {n_clips} clips due to missing reference data'
            )
            dataset = ClipCollection(
                ids=[c for c in dataset.ids if not c in remove]
            )

        # Filter clips that are excluded anyway
        n_clips = len(dataset.ids)
        exclude = set(amass.EXCLUDE)
        remove = set((clip for clip in dataset.ids if clip in exclude))
        if len(remove) > 0:
            log.info(f'Excluded {len(remove)} of {n_clips} clips')
            dataset = ClipCollection(
                ids=[c for c in dataset.ids if not c in remove]
            )

        if sample_by:
            counts: Dict[str, float] = defaultdict(float)
            for id in dataset.ids:
                md = metadata.get(id, {})
                counts[md.get(sample_by, '')] += float(md['duration'])
                if md.get(sample_by, '') == '':
                    log.debug(f'No entry for clip {id}')
            log.info(f'Sampling distribution: {counts}')
            weights = []
            total_duration = sum(counts.values())
            for id in dataset.ids:
                md = metadata.get(id, {})
                key = md.get(sample_by, '')
                weights.append(total_duration / counts[key])
            dataset.weights = tuple(weights)

        if n_ref < 1:
            ref_steps = [0]
            self.include_ref = False
        else:
            if past_ref:
                ref_steps = list(range(-n_ref, n_ref + 1))
            else:
                ref_steps = list(range(ref_offset, ref_offset + n_ref))
            self.include_ref = True

        if th.cuda.is_available() and not device.startswith('cuda'):
            raise RuntimeError(f'Expected cuda device, got {device}')
        device_id = 0
        try:
            device_id = int(device.split(':')[1])
        except:
            pass

        task = tracking.MultiClipMocapTracking(
            walker=walker_type,
            arena=arena,
            ref_path=ref_path,
            dataset=dataset,
            ref_steps=ref_steps,
            min_steps=min_steps,
            reward_type='comic',
            ghost_offset=[1, 0, 0] if ghost else None,
            always_init_at_clip_start=always_init_at_clip_start,
        )
        self.task = task

        # Assemble observables
        if features is None:
            features = {
                'observation': ['proprioceptive'],
                'time': ['time_in_clip'],
                'clip_id': ['clip_id'],
            }
        observables: Dict[str, List[str]] = defaultdict(list)
        for k, feats in features.items():
            if not feats:
                continue
            for f in feats:
                if f == 'proprioceptive':
                    observables[k] += _OBS_PROPRIO
                elif f == 'comic_ref':
                    observables[k] += _OBS_COMIC_REF
                else:
                    observables[k].append(f)
        if ref_lag > 1:
            observables['observation'].append('reference_lag_offset')

        self.env = hucc_cpp.BatchedMocapEnv(
            task=task,
            time_limit=time_limit,
            num_envs=n_envs,
            device=device_id,
            observables=dict(observables),
            verbose=verbose,
            reference_dims=reference_dims,
            reference_lag=ref_lag,
        )
        self.env.set_end_with_mocap(end_with_mocap)
        self.env.set_reward_type(reward_type)
        self.env.set_early_termination(early_termination)
        self.env.set_init_noise(init_noise)
        self.env.set_init_state(init_state)
        if init_offset is not None:
            self.env.set_init_offset(init_offset)

        self.action_space = gym.spaces.Box(
            low=self.env.action_spec()['minimum'].numpy(),
            high=self.env.action_spec()['maximum'].numpy(),
            dtype=np.float32,
        )
        self.env.reset()  # XXX needed to get observation spec at the moment
        obs_space = {}
        for k, v in self.env.observation_spec().items():
            if v.shape == (0,):
                continue
            if not self.include_ref and k == 'reference':
                continue
            elif k in ('clip_id', 'tick', 'frame', 'label'):
                obs_space[k] = gym.spaces.Discrete(int(v.max().item()))
            elif k == 'reference' and self.last_ref_only:
                assert len(v.shape) == 2
                obs_space[k] = box_space((v.shape[1],))
            else:
                obs_space[k] = box_space(v.shape)

        self.observation_space = gym.spaces.Dict(obs_space)
        self.ctx: Dict[str, Any] = {}

    def reference_data(self, keys):
        data = []
        for clip in self.task._all_clips:
            d = clip.as_dict()
            td = {}
            for k in keys:
                if k == 'body_height':
                    v = d['walker/position'][:, 2]
                else:
                    v = d[f'walker/{k}']
                td[k] = th.from_numpy(v).to(th.float32, copy=True)
            data.append(td)
        return data

    def set_clip(self, id: str):
        self.env.set_clip(id)

    def set_clip_features(self, data: Dict[str, th.Tensor]):
        self.env.set_clip_features(data)

    def clip_ids(self):
        return self.env.clip_ids()

    @property
    def dt(self):
        return 0.03

    @property
    def num_envs(self):
        return self.env.num_envs()

    def reset(self, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, list):
                seed = [seed]
            # truncate to 32bit, pybind complains
            self.env.seeds([s & 0xFFFFFFFF for s in seed])

        obs = self.env.reset()
        if self.last_ref_only:
            obs['reference'] = obs['reference'][:, -1]
        return obs, {}

    def reset_if_done(self):
        obs = self.env.reset_if_done()
        if self.last_ref_only:
            obs['reference'] = obs['reference'][:, -1]
        return obs, {}

    def step(self, action):
        obs, reward, done, reward_terms = self.env.step(action)
        info = {}
        info['rewards'] = {
            k.replace('_reward', ''): v for k, v in reward_terms.items()
        }
        if self.last_ref_only:
            obs['reference'] = obs['reference'][:, -1]
        terminated = done
        truncated = th.zeros_like(done)  # TODO propagate timeouts?
        return obs, reward, terminated, truncated, info

    def render_single(self, **kwargs):
        width = kwargs.get('width', 480)
        height = kwargs.get('height', 480)
        index = kwargs.get('index', 0)
        return self.env.render_single(width, height, index)
        # camera = kwargs.get('camera', 1)
        # flags = kwargs.get('flags', {})

    def render_all(self, **kwargs):
        width = kwargs.get('width', 480)
        height = kwargs.get('height', 480)
        return self.env.render_all(width, height)
