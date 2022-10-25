# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import operator
import sys
from collections import defaultdict
from itertools import accumulate
from os.path import dirname
from typing import Dict

import fairmotion
import numpy as np
import torch as th
from bisk.helpers import add_robot, root_with_floor
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.data import amass
from fairmotion.ops import conversions as fmc
from fairmotion.ops import quaternion
from fairmotion.utils import constants
from pytorch3d import transforms as p3t
from scipy.spatial.transform import Rotation

sys.path.insert(0, dirname(dirname(dirname(__file__))) + '/cpp/build')
try:
    import hucc_cpp_ext as hucc_cpp
except ModuleNotFoundError:
    hucc_cpp = None

# Defines how to joint rotations from AMASS, in Euler angles, to the MuJoCo
# joint positions (which are all just simple hinge joints)
JOINT_MAP = {
    'lfemurrx': ('lhip', 0, 'xyz'),
    'lfemurry': ('lhip', 1, 'xyz'),
    'lfemurrz': ('lhip', 2, 'xyz'),
    'rfemurrx': ('rhip', 0, 'xyz'),
    'rfemurry': ('rhip', 1, 'xyz'),
    'rfemurrz': ('rhip', 2, 'xyz'),
    'ltibiarx': ('lknee', 0, 'xyz'),
    'rtibiarx': ('rknee', 0, 'xyz'),
    'lfootrx': ('lankle', 0, 'xyz'),
    'lfootrz': ('lankle', 1, 'xyz'),
    'rfootrx': ('rankle', 0, 'xyz'),
    'rfootrz': ('rankle', 1, 'xyz'),
    'ltoesrx': ('ltoe', 0, 'xyz'),
    'rtoesrx': ('rtoe', 0, 'xyz'),
    'lowerbackrx': ('lowerback', 0, 'xyz'),
    'lowerbackry': ('lowerback', 1, 'xyz'),
    'lowerbackrz': ('lowerback', 2, 'xyz'),
    'upperbackrx': ('upperback', 0, 'xyz'),
    'upperbackry': ('upperback', 1, 'xyz'),
    'upperbackrz': ('upperback', 2, 'xyz'),
    'lowerneckrx': ('lowerneck', 0, 'xyz'),
    'lowerneckry': ('lowerneck', 1, 'xyz'),
    'lowerneckrz': ('lowerneck', 2, 'xyz'),
    'upperneckrx': ('upperneck', 0, 'xyz'),
    'upperneckry': ('upperneck', 1, 'xyz'),
    'upperneckrz': ('upperneck', 2, 'xyz'),
    'thoraxrx': ('chest', 0, 'xyz'),
    'thoraxry': ('chest', 1, 'xyz'),
    'thoraxrz': ('chest', 2, 'xyz'),
    'lclaviclery': ('lclavicle', 1, 'xyz'),
    'lclaviclerz': ('lclavicle', 2, 'xyz'),
    'rclaviclery': ('rclavicle', 1, 'xyz'),
    'rclaviclerz': ('rclavicle', 2, 'xyz'),
    'lhumerusrx': ('lshoulder', 1, 'xyz'),
    'lhumerusry': ('lshoulder', 0, 'xyz'),
    'lhumerusrz': ('lshoulder', 2, 'xyz'),
    'rhumerusrx': ('rshoulder', 1, 'xyz'),
    'rhumerusry': ('rshoulder', 0, 'xyz'),
    'rhumerusrz': ('rshoulder', 2, 'xyz'),
    'lradiusrx': ('lelbow', 0, 'yxz'),
    'rradiusrx': ('relbow', 0, 'yxz'),
    'lwristry': ('lwrist', 0, 'xyz'),
    'rwristry': ('rwrist', 0, 'xyz'),
    'lhandrx': ('lwrist', 0, 'yzx'),
    'lhandrz': ('lwrist', 1, 'yzx'),
    'rhandrx': ('rwrist', 0, 'yzx'),
    'rhandrz': ('rwrist', 1, 'yzx'),
}


def make_physics():
    '''
    Construct a MuJoCo physics instance for the AMASS-y Humanoid model.
    '''
    from dm_control import mjcf

    root = root_with_floor()
    _, robot_pos = add_robot(root, 'HumanoidAMASSPC', 'robot')
    p = mjcf.Physics.from_mjcf_model(root)
    return p


def pose_to_qpos(pose, p=None, prefix='robot') -> np.ndarray:
    if p is None:
        p = make_physics()

    orient, pos = fmc.T2Rp(pose.get_root_transform())

    p.named.data.qpos[f'{prefix}/'][0:3] = pos
    p.named.data.qpos[f'{prefix}/'][3:7] = fmc.R2Q(orient)[[3, 0, 1, 2]]

    for mj_joint, src in JOINT_MAP.items():
        src_joint, src_idx, src_order = src
        transform = pose.get_transform(
            pose.skel.get_joint(src_joint), local=True
        )
        r = fmc.R2E(fmc.T2R(transform), src_order)
        p.named.data.qpos[f'{prefix}/{mj_joint}'] = r[src_idx]

    return p.data.qpos.astype(np.float32, copy=True)


def qpos_to_pose(p, skel, prefix='robot'):
    eangles = {}
    for mj_joint, src in JOINT_MAP.items():
        src_joint, src_idx, src_order = src
        mj_pos = p.named.data.qpos[f'{prefix}/{mj_joint}']
        if not src_joint in eangles:
            eangles[src_joint] = defaultdict(int)
        eangles[src_joint][src_order[src_idx]] = mj_pos[0]
    pos = p.data.qpos[0:3]
    orient = p.data.qpos[3:7][[1, 2, 3, 0]]
    root_transform = fmc.Rp2T(fmc.Q2R(orient), pos)

    pose = fairmotion.core.motion.Pose(
        skel, data=[constants.eye_T() for _ in range(skel.num_joints())]
    )
    pose.set_root_transform(root_transform, local=True)
    for k, v in eangles.items():
        rot = fmc.E2R(np.array([v.get('x', 0), v.get('y', 0), v.get('z', 0)]))
        T = fmc.Rp2T(rot, np.zeros(3))
        pose.set_transform(skel.get_joint(k), T, local=True)
    return pose


def motion_to_qpos(motion, p=None) -> np.ndarray:
    '''
    Convert motion into a qpos array for the MuJoCo Humanoid.
    '''
    if p is None:
        p = make_physics()

    qpos = []
    for i in range(motion.num_frames()):
        qpos.append(pose_to_qpos(motion.get_pose_by_frame(i), p))
    return np.stack(qpos)


def motion_to_comic(motion, p=None) -> Dict[str, np.ndarray]:
    '''
    Featurize motion as in Hasenclever, 2020, "CoMic: Complementary Task Learning &
    Mimicry for Reusable Skills".
    Also returns qpos.
    '''
    from dm_control.mujoco import math as mjmath
    from dm_control.mujoco.wrapper.mjbindings import mjlib

    if p is None:
        p = make_physics()
    qpos = motion_to_qpos(motion, p)

    # Additional features
    appendages = []
    body_positions = []
    body_quaternions = []
    center_of_mass = []
    end_effectors = []
    joints = []
    position = []
    quaternion = []

    # TODO string indexing
    xpos_quat_idxs = list(range(3, 34))
    xp_names = p.named.data.xpos.axes.row.names
    end_effectors_idx = [
        xp_names.index(f'robot/{k}')
        for k in ('rradius', 'lradius', 'rfoot', 'lfoot')
    ]
    appendages_idx = [
        xp_names.index(f'robot/{k}')
        for k in ('rradius', 'lradius', 'rfoot', 'lfoot', 'head')
    ]

    for i in range(qpos.shape[0] - 1):
        p.data.qpos[:] = qpos[i]
        mjlib.mj_kinematics(p.model.ptr, p.data.ptr)
        mjlib.mj_comPos(p.model.ptr, p.data.ptr)

        appendages.append(
            np.matmul(
                p.data.xpos[appendages_idx] - p.data.xpos[2],
                p.data.xmat[2].reshape(3, 3),
            ).astype(np.float32)
        )
        body_positions.append(
            p.data.xpos[xpos_quat_idxs].astype(np.float32, copy=True)
        )
        body_quaternions.append(
            p.data.xquat[xpos_quat_idxs].astype(np.float32, copy=True)
        )
        center_of_mass.append(
            p.data.subtree_com[1].astype(np.float32, copy=True)
        )
        end_effectors.append(
            np.matmul(
                p.data.xpos[end_effectors_idx] - p.data.xpos[2],
                p.data.xmat[2].reshape(3, 3),
            ).astype(np.float32)
        )
        joints.append(p.data.qpos[7:63].astype(np.float32, copy=True))
        position.append(p.data.qpos[0:3].astype(np.float32, copy=True))
        quaternion.append(p.data.qpos[3:7].astype(np.float32, copy=True))

    # Compute velocity via finite differences
    # cf. dm_control.suite.utils.parse_amc.convert()
    velocity = []
    angular_velocity = []
    joints_velocity = []
    for i in range(qpos.shape[0] - 1):
        p_t = qpos[i]
        p_tp1 = qpos[i + 1]
        velocity.append((p_tp1[:3] - p_t[:3]) * motion.fps)
        angular_velocity.append(
            mjmath.mj_quat2vel(
                mjmath.mj_quatdiff(
                    p_t[3:7].astype(np.float64), p_tp1[3:7].astype(np.float64)
                ),
                1 / motion.fps,
            )
        )
        joints_velocity.append((p_tp1[7:63] - p_t[7:63]) * motion.fps)

    n = qpos.shape[0] - 1
    return {
        'qpos': qpos[:-1],
        'appendages': np.stack(appendages).reshape(n, -1),
        'body_positions': np.stack(body_positions).reshape(n, -1),
        'body_quaternions': np.stack(body_quaternions).reshape(n, -1),
        'center_of_mass': np.stack(center_of_mass).reshape(n, -1),
        'end_effectors': np.stack(end_effectors).reshape(n, -1),
        'joints': np.stack(joints).reshape(n, -1),
        'position': np.stack(position).reshape(n, -1),
        'quaternion': np.stack(quaternion).reshape(n, -1),
        'velocity': np.stack(velocity).reshape(n, -1),
        'angular_velocity': np.stack(angular_velocity).reshape(n, -1),
        'joints_velocity': np.stack(joints_velocity).reshape(n, -1),
    }


def R2FU(R):
    fwdup = np.matmul(R, np.array([[1, 0, 0], [0, 0, 1]]).T)
    fwd, up = np.split(fwdup, 2, -1)
    return fwd.squeeze(-1), up.squeeze(-1)


def FU2R(fwd, up):
    x = fwd.reshape(-1, 3)
    # Make up-vector orthogonal to forward
    z = up.reshape(-1, 3) - x * (
        np.einsum('bx,bx->b', x, up.reshape(-1, 3))
        / np.einsum('bx,bx->b', x, x)
    ).reshape(-1, 1)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1)
    if fwd.shape[-1] == 3:
        R = R.reshape((fwd.shape[:-1]) + (3, 3))
    return R


def R2R6(R):
    return np.concatenate([R.take(0, -1), R.take(1, -1)], axis=-1)


def R62R(A):
    a1 = np.atleast_2d(A.take(range(3), -1))
    a2 = np.atleast_2d(A.take(range(3, 6), -1))
    c1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    c2 = a2 - (np.expand_dims(np.einsum('...bx,...bx->...b', c1, a2), -1) * c1)
    c2 = c2 / np.linalg.norm(c2, axis=-1, keepdims=True)
    c3 = np.cross(c1, c2)
    R = np.stack([c1, c2, c3], axis=-1)
    if A.ndim == 1:
        R = R[0]
    return R


def motion_to_jposrot(motion) -> Dict[str, np.ndarray]:
    '''
    Featurize motion as (global) joint positions and rotations.
    Rotations are expressed as forward (X) and upward (Z) vectors as in
    Ling, 2020, "Character controllers using motion VAEs"
    '''
    if hucc_cpp is None:
        raise RuntimeError('C++ extensions could not be loaded')

    # XXX Assumes that motion uses the AMASS skeleton
    num_joints = len(amass.joint_names)

    jpos = motion.positions(local=False)
    rots = motion.rotations(local=True)
    fwd, up = R2FU(rots)
    r6 = R2R6(rots)

    # Root quaternion diffs
    rquatf = fmc.R2Q(rots[:, 0]).astype(np.float32)
    bquatf = fmc.R2Q(rots[:, 1:])
    # Come on scipy, who puts w at the end?
    bquat = bquatf[:, :, [3, 0, 1, 2]].reshape(bquatf.shape[0], -1)
    rquat = rquatf[:, [3, 0, 1, 2]]
    rqd = hucc_cpp.quat_diff(
        th.tensor(rquat[:-1]), th.tensor(rquat[1:])
    ).numpy()
    rquatr = np.concatenate(
        [np.array([1, 0, 0, 0]).astype(np.float32).reshape(-1, 4), rqd]
    )
    rrelr6 = np.concatenate(
        [
            R2R6(fmc.Q2R(np.array([1, 0, 0, 0])))
            .astype(np.float32)
            .reshape((-1, 6)),
            R2R6(fmc.Q2R(rqd[:, [1, 2, 3, 0]])),
        ]
    )

    vmotion = MotionWithVelocity.from_motion(motion)
    avels = []
    lvels = []
    for i in range(motion.num_frames()):
        vels = vmotion.get_velocity_by_frame(i).data_local  # XXX to_matrix()
        avels.append(vels[:, 0:3])
        lvels.append(vels[:, 3:6])
    avel = np.concatenate(avels).reshape(-1, 3 * num_joints)
    lvel = np.concatenate(lvels).reshape(-1, 3 * num_joints)

    jpos = jpos.reshape(-1, 3 * num_joints)
    fwd = fwd.reshape(-1, 3 * num_joints)
    up = up.reshape(-1, 3 * num_joints)
    r6 = r6.reshape(-1, 6 * num_joints)
    br6 = r6[:, 6:]  # without root joint
    rpos = jpos[:, 0:3]
    rvel = lvel[:, 0:3]
    relpos = np.concatenate([np.zeros((1, 3)), rpos[1:] - rpos[:-1]])
    relxypos = np.concatenate(
        [
            np.concatenate([np.zeros((1, 2)), rpos[1:, :2] - rpos[:-1, :2]]),
            rpos[:, 2:3],
        ],
        axis=-1,
    )

    # Local root veclocity (m/dt)
    rlvel = p3t.quaternion_apply(th.tensor(rquat), th.tensor(relpos)).numpy()

    return {
        'jpos': jpos.astype(np.float32),
        'rpos': rpos.astype(np.float32),  # root position
        'relpos': relpos.astype(np.float32),
        'relxypos': relxypos.astype(np.float32),
        'fwd': fwd.astype(np.float32),
        'up': up.astype(np.float32),
        'r6': r6.astype(np.float32),
        'rvel': rvel.astype(np.float32),
        'avel': avel.astype(np.float32),
        'lvel': lvel.astype(np.float32),
        'br6': br6.astype(np.float32),
        'rrelr6': rrelr6.astype(np.float32),
        'rquat': rquat.astype(np.float32),
        'rquatr': rquatr.astype(np.float32),
        'bquat': bquat.astype(np.float32),
        'rlvel': rlvel.astype(np.float32),
    }


def jposrot_to_qpos(data, p=None):
    if p is None:
        p = make_physics()

    if 'r6' in data:
        R = R62R(data['r6'].reshape(-1, 6))
    else:
        R = FU2R(data['fwd'], data['up'])

    if 'jpos' in data:
        p.data.qpos[0:3] = data['jpos'][0:3]
    elif 'rpos' in data:
        p.data.qpos[0:3] = data['rpos']
    else:
        raise ValueError(
            f'No position information in data: {list(data.keys())}'
        )
    p.data.qpos[3:7] = fmc.R2Q(R[0])[[3, 0, 1, 2]]
    for mj_joint, src in JOINT_MAP.items():
        src_joint, src_idx, src_order = src
        r = fmc.R2E(R[amass.joint_names.index(src_joint)], src_order)
        p.named.data.qpos[f'robot/{mj_joint}'] = r[src_idx]

    return p.data.qpos.astype(np.float32, copy=True)


def jposrot_to_motion(data, skel, fps=1 / 0.03, initial_pose=None):
    data = dict(data)
    if not 'rpos' in data and 'relxypos' in data:
        data['rpos'] = data['relxypos'].copy()
        if initial_pose is not None:
            sposxy = fmc.T2p(initial_pose.get_root_transform())[:2]
            data['rpos'][0, :2] = sposxy
        data['rpos'][:, :2] = np.cumsum(data['rpos'][:, :2], axis=0)

    if 'rrelr6' in data and 'br6' in data and not 'r6' in data:
        Rrd = Rotation.from_matrix(R62R(data['rrelr6']))
        if initial_pose is not None:
            Rrd[0] = Rotation.from_matrix(
                fmc.T2R(initial_pose.get_root_transform())
            )
        Rr = np.stack([r.as_matrix() for r in accumulate(Rrd, operator.mul)])
        Rb = R62R(data['br6'].reshape(-1, 6)).reshape(-1, 21, 3, 3)
        R = np.concatenate([Rr[:, np.newaxis], Rb], axis=1)
    elif 'rquatr' in data and not 'r6' in data:
        Rrd = Rotation.from_quat(data['rquatr'][:, [1, 2, 3, 0]])
        if initial_pose is not None:
            Rrd[0] = Rotation.from_matrix(
                fmc.T2R(initial_pose.get_root_transform())
            )
        Rr = np.stack([r.as_matrix() for r in accumulate(Rrd, operator.mul)])
        # Rr = Rrd.as_matrix()
        if 'br6' in data:
            Rb = R62R(data['br6'].reshape(-1, 6)).reshape(-1, 21, 3, 3)
        elif 'bquat' in data:
            Rb = (
                Rotation.from_quat(
                    data['bquat'].reshape(-1, 4)[:, [1, 2, 3, 0]]
                )
                .as_matrix()
                .reshape(-1, 21, 3, 3)
            )
        else:
            raise NotImplementedError()
        R = np.concatenate([Rr[:, np.newaxis], Rb], axis=1)
    elif 'r6' in data:
        R = R62R(data['r6'].reshape(-1, 6)).reshape(-1, 22, 3, 3)
    else:
        R = FU2R(data['fwd'], data['up'])

    if not 'rpos' in data and 'rlvel' in data:
        relpos = Rotation.from_matrix(R[:, 0]).inv().apply(data['rlvel'])
        if initial_pose is not None:
            relpos[0] = fmc.T2p(initial_pose.get_root_transform())
        data['rpos'] = np.cumsum(relpos, axis=0)

    T = fmc.R2T(R)
    T[:, 0, :3, 3] = data['rpos']
    return fairmotion.core.motion.Motion.from_matrix(
        T, skel=skel, local=True, fps=fps
    )


def rotate_motion_z(motion, angle_deg: float):
    R = fmc.Az2R(fmc.deg2rad(angle_deg))
    # Rotate all joints
    # TODO fairmotion.ops.motion.rotate is broken on master? Fix!
    # actually I think it does the full job now?
    motion = fairmotion.ops.motion.rotate(motion, R)
    '''
    # Rotate all positions
    for i in range(len(motion.poses)):
        R0, p0 = fmc.T2Rp(motion.poses[i].get_root_transform())
        p = np.dot(R, p0)
        motion.poses[i].set_root_transform(fmc.Rp2T(R0, p), local=False)
    '''
    return motion
