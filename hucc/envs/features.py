# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import gym
import numpy as np
from bisk.features import Featurizer
from bisk.features.joints import JointsRelZFeaturizer
from dm_control.utils.transformations import (quat_diff, quat_to_axisangle,
                                              quat_to_mat)
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)


class BodyFeetWalkerFeaturizer(Featurizer):
    def __init__(self, p, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        assert robot == 'walker', f'Walker robot expected, got "{robot}"'
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        root = self.p.named.data.qpos[[f'{self.prefix}/root{p}' for p in 'zxy']]
        torso_frame = self.p.named.data.xmat[f'{self.prefix}/torso'].reshape(
            3, 3
        )
        torso_pos = self.p.named.data.xpos[f'{self.prefix}/torso']
        positions = []
        for side in ('left', 'right'):
            torso_to_limb = (
                self.p.named.data.xpos[f'{self.prefix}/{side}_foot'] - torso_pos
            )
            # We're in 2D effectively, y is constant
            positions.append(torso_to_limb.dot(torso_frame)[[0, 2]])
        extremities = np.hstack(positions)
        return np.concatenate([root, extremities], dtype=np.float32)

    def feature_names(self) -> List[str]:
        names = ['rootz:p', 'rootx:p', 'rooty:p']
        names += [f'left_foot:p{p}' for p in 'xz']
        names += [f'right_foot:p{p}' for p in 'xz']
        return names


class BodyFeetRelZWalkerFeaturizer(BodyFeetWalkerFeaturizer):
    def __init__(self, p, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[0] = self.relz()
        return obs


class BodyFeetHumanoidFeaturizer(Featurizer):
    def __init__(
        self, p, robot: str, prefix: str = 'robot', exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)
        self.for_pos = None
        self.for_twist = None
        self.foot_anchor = 'pelvis'
        self.reference = 'torso'
        self.limbs = ['left_foot', 'right_foot']
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    @staticmethod
    def decompose_twist_swing_z(q):
        p = [0.0, 0.0, q[2]]
        twist = Rotation.from_quat(np.array([p[0], p[1], p[2], q[3]]))
        swing = Rotation.from_quat(q) * twist.inv()
        return twist, swing

    def __call__(self) -> np.ndarray:
        root = self.p.named.data.xpos[f'{self.prefix}/{self.reference}']
        if self.for_pos is not None:
            root = root.copy()
            root[0:2] -= self.for_pos
            root[0:2] = self.for_twist.apply(root * np.array([1, 1, 0]))[0:2]
        q = self.p.data.qpos[3:7]
        t, s = self.decompose_twist_swing_z(q[[1, 2, 3, 0]])
        tz = t.as_rotvec()[2]
        e = s.as_euler('yzx')
        sy, sx = e[0], e[2]

        # Feet positions are relative to pelvis position and its heading
        # Also, exclude hands for now.
        pelvis_q = self.p.named.data.xquat[f'{self.prefix}/{self.foot_anchor}']
        pelvis_t, pelvis_s = self.decompose_twist_swing_z(
            pelvis_q[[1, 2, 3, 0]]
        )
        pelvis_pos = self.p.named.data.xpos[f'{self.prefix}/{self.foot_anchor}']
        positions = []
        for limb in self.limbs:
            pelvis_to_limb = (
                self.p.named.data.xpos[f'{self.prefix}/{limb}'] - pelvis_pos
            )
            positions.append(pelvis_t.apply(pelvis_to_limb))
        extremities = np.hstack(positions)
        return np.concatenate(
            [root, np.asarray([tz, sy, sx]), extremities], dtype=np.float32
        )

    def feature_names(self) -> List[str]:
        names = [f'root:p{f}' for f in 'xyz']
        names += [f'root:t{f}' for f in 'z']
        names += [f'root:s{f}' for f in 'yx']
        for limb in self.limbs:
            names += [f'{limb}:p{f}' for f in 'xyz']
        return names


class BodyFeetRelZHumanoidFeaturizer(BodyFeetHumanoidFeaturizer):
    def __init__(self, p, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[2] = self.relz()
        return obs


class BodyFeetHumanoidAMASSFeaturizer(BodyFeetHumanoidFeaturizer):
    def __init__(self, p, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        self.foot_anchor = 'torso'
        self.reference = 'lowerneck'
        self.limbs = ['lfoot', 'rfoot']


class ComicProprioFeaturizer(Featurizer):
    def __init__(
        self, p, robot: str, prefix: str = 'robot', exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)

        self.qpos_idx = []
        self.qvel_idx = []
        for jn in self.p.named.model.jnt_type.axes.row.names:
            if not jn.startswith(f'{self.prefix}/'):
                continue
            if exclude is not None and re.match(exclude, jn) is not None:
                continue
            typ = self.p.named.model.jnt_type[jn]
            qpos_adr = self.p.named.model.jnt_qposadr[jn]
            for i in range(self.n_qpos[typ]):
                self.qpos_idx.append(qpos_adr + i)
            qvel_adr = self.p.named.model.jnt_dofadr[jn]
            for i in range(self.n_qvel[typ]):
                self.qvel_idx.append(qvel_adr + i)

        xpos_names = p.named.data.xpos.axes.row.names
        self.end_effectors_idx = [
            xpos_names.index(f'{self.prefix}/rradius'),
            xpos_names.index(f'{self.prefix}/lradius'),
            xpos_names.index(f'{self.prefix}/rfoot'),
            xpos_names.index(f'{self.prefix}/lfoot'),
        ]
        self.head_idx = [xpos_names.index(f'{self.prefix}/head')]

        self.sensors_idx = defaultdict(list)
        ax = p.named.data.sensordata.axes[0]
        self.sensors_idx['veloc'] = ax.convert_key_item(
            f'{self.prefix}/sensor_root_veloc'
        )
        self.sensors_idx['gyro'] = ax.convert_key_item(
            f'{self.prefix}/sensor_root_gyro'
        )
        self.sensors_idx['accel'] = ax.convert_key_item(
            f'{self.prefix}/sensor_root_accel'
        )
        for sn in p.named.model.sensor_adr.axes.row.names:
            idx = p.named.model.sensor_adr[sn]
            n = p.named.model.sensor_dim[sn]
            if sn.startswith(f'{self.prefix}/sensor_touch'):
                self.sensors_idx['touch'] += list(range(idx, idx + n))
            elif sn.startswith(f'{self.prefix}/sensor_torque'):
                self.sensors_idx['torque'] += list(range(idx, idx + n))

        self.prev: Dict[str, np.ndarray] = {}
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(286,), dtype=np.float32
        )

        # XXX hard-coded from mocap environment
        # fmt: off
        self.mocap_to_observable_joint_order = [31, 30, 29, 33, 32, 2, 1, 0, 41,
                5, 4, 40, 39, 36, 35, 34, 16, 15, 14, 25, 24, 23, 37, 43, 42, 3,
                6, 38, 45, 44, 9, 8, 7, 53, 12, 11, 52, 51, 48, 47, 46, 49, 55,
                54, 10, 13, 50, 22, 21, 20, 19, 18, 17, 28, 27, 26,]
        # fmt: on

    def reset(self):
        d = self.p.data
        quaternion = d.qpos[self.qpos_idx][3:7]
        position = d.qpos[self.qpos_idx][0:3]
        joints_pos = d.qpos[self.qpos_idx][7:]
        self.prev = {
            'quaternion': quaternion.copy(),
            'position': position.copy(),
            'joints_pos': joints_pos.copy(),
        }

    def __call__(self) -> np.ndarray:
        _TORQUE_THRESHOLD = 60
        _TOUCH_THRESHOLD = 1e-3

        d = self.p.data
        nd = self.p.named.data
        actuator_activation = d.act
        position = d.qpos[self.qpos_idx][0:3]
        quaternion = d.qpos[self.qpos_idx][3:7]
        joints_pos = d.qpos[self.qpos_idx][7:]
        joints_vel = d.qvel[self.qvel_idx][6:]

        appendages_pos = np.matmul(
            d.xpos[self.end_effectors_idx + self.head_idx]
            - nd.xpos[f'{self.prefix}/torso'],
            nd.xmat[f'{self.prefix}/torso'].reshape(3, 3),
        )
        end_effectors_pos = np.matmul(
            d.xpos[self.end_effectors_idx] - nd.xpos[f'{self.prefix}/torso'],
            nd.xmat[f'{self.prefix}/torso'].reshape(3, 3),
        )
        body_height = nd.xpos[f'{self.prefix}/torso', 'z']
        world_zaxis = nd.xmat[f'{self.prefix}/torso'][6:9]
        sensors_velocimeter = d.sensordata[self.sensors_idx['veloc']]
        sensors_gyro = d.sensordata[self.sensors_idx['gyro']]
        sensors_accelerometer = d.sensordata[self.sensors_idx['accel']]
        sensors_touch = (
            d.sensordata[self.sensors_idx['touch']] > _TOUCH_THRESHOLD
        )
        sensors_torque = np.tanh(
            2 * d.sensordata[self.sensors_idx['torque']] / _TORQUE_THRESHOLD
        )

        control_timestep = (
            self.p.timestep() * 6
        )  # XXX control_timestep_ -> how to get it?

        gyro_control = quat_diff(
            self.prev.get('quaternion', quaternion), quaternion
        )
        gyro_control = quat_to_axisangle(
            gyro_control / np.linalg.norm(gyro_control)
        )
        gyro_control /= control_timestep

        joints_vel_control = (
            joints_pos - self.prev.get('joints_pos', joints_pos)
        ) / control_timestep
        joints_vel_control = joints_vel_control[
            self.mocap_to_observable_joint_order
        ]

        rmat_prev = quat_to_mat(self.prev.get('quaternion', quaternion))[:3, :3]
        velocimeter_control = (
            position - self.prev.get('position', position)
        ) / control_timestep
        velocimeter_control = np.matmul(velocimeter_control, rmat_prev)

        ret = np.concatenate(
            [
                appendages_pos.reshape(-1),
                body_height.reshape(-1),
                joints_pos.reshape(-1),
                joints_vel.reshape(-1),
                gyro_control.reshape(-1),
                joints_vel_control.reshape(-1),
                velocimeter_control.reshape(-1),
                sensors_touch.reshape(-1),
                sensors_velocimeter.reshape(-1),
                sensors_gyro.reshape(-1),
                sensors_accelerometer.reshape(-1),
                end_effectors_pos.reshape(-1),
                actuator_activation.reshape(-1),
                sensors_torque.reshape(-1),
                world_zaxis.reshape(-1),
            ],
            dtype=np.float32,
        )

        self.prev = {
            'quaternion': quaternion.copy(),
            'position': position.copy(),
            'joints_pos': joints_pos.copy(),
        }

        return ret

    def feature_names(self) -> List[str]:
        # TODO proper names?
        names = []
        names += [f'appendages_pos:{f}' for f in range(15)]
        names += [f'body_height:{f}' for f in range(1)]
        names += [f'joints_pos:{f}' for f in range(56)]
        names += [f'joints_vel:{f}' for f in range(56)]
        names += [f'gyro_control:{f}' for f in range(3)]
        names += [f'joints_vel_control:{f}' for f in range(56)]
        names += [f'velocimeter_control:{f}' for f in range(3)]
        names += [f'sensors_touch:{f}' for f in range(10)]
        names += [f'sensors_velocimeter:{f}' for f in range(3)]
        names += [f'sensors_gyro:{f}' for f in range(3)]
        names += [f'sensors_accelerometer:{f}' for f in range(3)]
        names += [f'end_effectors_pos:{f}' for f in range(12)]
        names += [f'actuator_activation:{f}' for f in range(56)]
        names += [f'sensors_torque:{f}' for f in range(6)]
        names += [f'world_zaxis:{f}' for f in range(3)]
        return names


class ComicFeaturizer(ComicProprioFeaturizer):
    def __init__(
        self, p, robot: str, prefix: str = 'robot', exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)

        nobs = self.observation_space.shape[0] + 12
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(nobs,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        d = self.p.data
        rqposp = d.qpos[self.qpos_idx][:2]
        rqposq = d.qpos[self.qpos_idx][3:7]
        rqvel = d.qpos[self.qpos_idx][:6]
        return np.concatenate(
            [obs, rqposp.reshape(-1), rqposq.reshape(-1), rqvel.reshape(-1)],
            dtype=np.float32,
        )

    def feature_names(self):
        names = super().feature_names()
        names += [f'qpos:{f}' for f in range(2)]
        names += [f'qpos:{f}' for f in range(3, 7)]
        names += [f'qvel:{f}' for f in range(6)]
        return names


class ComicRelZFeaturizer(ComicFeaturizer):
    def __init__(
        self, p, robot: str, prefix: str = 'robot', exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[15] = self.relz()
        return obs


class ComicProprioRelZFeaturizer(ComicProprioFeaturizer):
    def __init__(
        self, p, robot: str, prefix: str = 'robot', exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[15] = self.relz()
        return obs


# In pre-training, we may pass a propioceptive deltas to the robot -- this
# featurizer just sets them to zero.
class ComicProprioZeroDeltaFeaturizer(ComicProprioFeaturizer):
    def __init__(
        self,
        num_zeros: int,
        p,
        robot: str,
        prefix: str = 'robot',
        exclude: str = None,
    ):
        super().__init__(p, robot, prefix, exclude)
        self.num_zeros = num_zeros
        nobs = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(nobs + self.num_zeros,),
            dtype=np.float32,
        )

    def __call__(self) -> np.ndarray:
        ret = super().__call__()
        return np.concatenate([ret, np.zeros(self.num_zeros)], dtype=np.float32)

    def feature_names(self) -> List[str]:
        names = super().feature_names()
        for i in range(self.num_zeros):
            names.append(f'zero_{i}')
        return names


class ComicProprioZeroDeltaRelZFeaturizer(ComicProprioZeroDeltaFeaturizer):
    def __init__(
        self,
        num_zeros: int,
        p,
        robot: str,
        prefix: str = 'robot',
        exclude: str = None,
    ):
        super().__init__(num_zeros, p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[15] = self.relz()
        return obs


class UngroupFeatures(gym.ObservationWrapper):
    '''
    Observation wrapper that ungroups bisk features based on feature names.
    '''

    def __init__(self, env, prefix: str = None):
        from bisk import BiskSingleRobotEnv

        super().__init__(env)
        assert isinstance(
            env.unwrapped, BiskSingleRobotEnv
        ), 'BiskFeatures requires a BiskSingleRobotEnv environment'

        self.prefix = prefix
        fname = lambda fn: f'{prefix}{fn}' if prefix else fn

        self.ranges: Dict[str, Tuple[int, int]] = {}
        fnames = env.featurizer.feature_names()
        last, start = None, 0
        for i, fn in enumerate(fnames):
            group = fn.split(':')[0]
            if group != last:
                if last is not None:
                    assert not last in self.ranges
                    self.ranges[fname(last)] = (start, i)
                start = i
                last = group
        if last is not None:
            assert not last in self.ranges
            self.ranges[fname(last)] = (start, i)

        # Let's not care about ranges for now
        spaces = {
            f'{prefix}{k}': v
            for k, v in env.observation_space.spaces.items()
            if k != 'observation'
        }
        spaces.update(
            {
                k: gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(v[1] - v[0],),
                    dtype=np.float32,
                )
                for k, v in self.ranges.items()
            }
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        uobs = {
            f'{self.prefix}{k}': v
            for k, v in observation.items()
            if k != 'observation'
        }
        obs = observation['observation']
        for k, (start, end) in self.ranges.items():
            uobs[k] = obs[start:end]
        return uobs


def bodyfeet_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'walker':
        return BodyFeetWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoid' or robot == 'humanoidpc':
        return BodyFeetHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoidamasspc':
        return BodyFeetHumanoidAMASSFeaturizer(
            p, robot, prefix, *args, **kwargs
        )
    else:
        raise ValueError(f'No bodyfeet featurizer for robot "{robot}"')


def bodyfeet_relz_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'walker':
        return BodyFeetRelZWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoid' or robot == 'humanoidpc':
        return BodyFeetRelZHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoidamasspc':
        return BodyFeetRelZHumanoidAMASSFeaturizer(
            p, robot, prefix, *args, **kwargs
        )
    else:
        raise ValueError(f'No bodyfeet-relz featurizer for robot "{robot}"')


def comic_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No comic featurizer for robot "{robot}"')


def comic_relz_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicRelZFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No comic featurizer for robot "{robot}"')


def comic_proprio_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicProprioFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No comic proprio featurizer for robot "{robot}"')


def comic_proprio_relz_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicProprioRelZFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No comic proprio featurizer for robot "{robot}"')


def comic_propriozd_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicProprioZeroDeltaFeaturizer(
            2, p, robot, prefix, *args, **kwargs
        )
    else:
        raise ValueError(f'No comic proprio featurizer for robot "{robot}"')


def comic_propriozd_relz_featurizer(
    p, robot: str, prefix: str, *args, **kwargs
):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicProprioZeroDeltaRelZFeaturizer(
            2, p, robot, prefix, *args, **kwargs
        )
    else:
        raise ValueError(f'No comic proprio featurizer for robot "{robot}"')


def comic_propriozd6_featurizer(p, robot: str, prefix: str, *args, **kwargs):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicProprioZeroDeltaFeaturizer(
            6, p, robot, prefix, *args, **kwargs
        )
    else:
        raise ValueError(f'No comic proprio featurizer for robot "{robot}"')


def comic_propriozd6_relz_featurizer(
    p, robot: str, prefix: str, *args, **kwargs
):
    if robot == 'humanoidcmupc' or robot == 'humanoidamasspc':
        return ComicProprioZeroDeltaRelZFeaturizer(
            6, p, robot, prefix, *args, **kwargs
        )
    else:
        raise ValueError(f'No comic proprio featurizer for robot "{robot}"')


if __name__ == '__main__':
    import ctypes

    import numpy as np
    import torch as th
    from dm_control.mujoco import engine, wrapper
    from dm_control.mujoco.wrapper import mjbindings

    import hucc

    # Test comic features
    env = gym.make(
        'CMUMocap2020CppMulti-v1',
        robot='AMASS',
        ref_path='/scratch/jgehring/data/amass-mjbox/amass-ground-truth-1.h5',
        dataset='comic_run_jump_tiny',
        reference_dims=2,
        verbose=False,
        ref_features=['ref_relxypos', 'ref_r6'],
    )

    def rewrap_model(model):
        return wrapper.MjModel(
            ctypes.cast(model, ctypes.POINTER(mjbindings.types.MJMODEL))
        )

    def rewrap_data(data, model):
        r = wrapper.MjData(model)
        r._ptr = ctypes.cast(data, ctypes.POINTER(mjbindings.types.MJDATA))
        return r

    model = rewrap_model(env.env.model()[0])
    data = rewrap_data(env.env.data()[0], model)
    p = engine.Physics(data)
    f = ComicProprioFeaturizer(p, 'humanoidamasspc', prefix='walker')

    print('reset')
    env.reset()
    f.reset()
    for i in range(200):
        obs, reward, done, _ = env.step(
            th.from_numpy(env.action_space.sample())
        )
        obs1 = obs['observation'].cpu()[0]
        obs2 = th.from_numpy(f().astype(np.float32))
        assert th.allclose(obs1, obs2)
        if done:
            print('reset')
            env.reset()
            f.reset()
