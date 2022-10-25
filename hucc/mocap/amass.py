# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import sys
from concurrent import futures
from os import path as osp
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch as th
from fairmotion.core import motion as motion_class
from fairmotion.data import amass
from fairmotion.ops import conversions as fmc
from fairmotion.ops.motion import resample
from human_body_prior.body_model.body_model import BodyModel

from hucc.mocap.envs import AMASSReplayEnv

log = logging.getLogger(__name__)
NUM_BETAS = 16


class AMassMotion(motion_class.Motion):
    def __init__(self, gender, betas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gender = gender
        self._betas = betas

    @property
    def gender(self):
        return self._gender

    @property
    def betas(self):
        return self._betas


class BodyModels(SimpleNamespace):
    female: BodyModel
    male: BodyModel

    def keys(self):
        return ['female', 'male']

    def __getitem__(self, key):
        return getattr(self, key)


def load_amass_bms(smplh_path: str):
    return BodyModels(
        male=BodyModel(
            bm_fname=osp.join(smplh_path, 'male/model.npz'), num_betas=NUM_BETAS
        ).to('cpu'),
        female=BodyModel(
            bm_fname=osp.join(smplh_path, 'female/model.npz'),
            num_betas=NUM_BETAS,
        ).to('cpu'),
    )


def motion_to_amass(motion: AMassMotion, clip_path: str):
    R, p = fmc.T2Rp(motion.to_matrix())
    trans = p[:, 0].copy()

    A = fmc.R2A(R).reshape(R.shape[0], -1)
    # XXX no rotation data for hands available at this point -- fill with zeros
    poses = np.concatenate(
        [A, np.zeros((A.shape[0], 156 - A.shape[1]))], axis=1
    )

    np.savez(
        clip_path,
        gender=motion.gender,
        betas=motion.betas,
        mocap_framerate=motion.fps,
        trans=trans,
        poses=poses,
    )


def amass_to_motion(
    clip_path: str,
    bms: BodyModels,
    betas: Dict[str, np.ndarray] = None,
    fps: float = None,
):
    data = np.load(clip_path)
    orig_fps = float(data['mocap_framerate'])
    try:
        gender = data['gender'].decode('ascii')
    except:
        gender = str(data['gender'])
    try:
        # XXX too lazy to debug...
        gender = gender.replace("b'", "")
        gender = gender.replace("'", "")
    except:
        pass

    if betas is None:
        skel_betas = th.tensor(data['betas'], dtype=th.float32).view(
            1, NUM_BETAS
        )
    else:
        skel_betas = (
            th.from_numpy(betas[gender]).to(th.float32).view(1, NUM_BETAS)
        )

    skel = amass.create_skeleton_from_amass_bodymodel(
        bms[gender], skel_betas, len(amass.joint_names), amass.joint_names
    )
    motion = AMassMotion(
        gender=gender, betas=data['betas'], skel=skel, fps=orig_fps
    )

    root_orient = data['poses'][:, :3]  # controls the global root orientation
    pose_body = data['poses'][
        :, 3 : len(amass.joint_names) * 3
    ]  # controls body joint angles
    trans = data['trans'][:, :3]

    num_joints = skel.num_joints()
    parents = bms[gender].kintree_table[0].long()[:num_joints]

    if betas is not None:
        # Estimate Z offset for custom betas for zero pose
        pose_body_zeros = th.zeros((1, 3 * (num_joints - 1)))
        body_o = bms[gender](
            pose_body=pose_body_zeros,
            trans=th.zeros((1, 3)),
            root_orient=th.zeros((1, 3)),
            betas=skel_betas,
        )
        body = bms[gender](
            pose_body=pose_body_zeros,
            trans=th.zeros((1, 3)),
            root_orient=th.zeros((1, 3)),
            betas=th.from_numpy(data['betas']).view(1, -1).to(th.float32),
        )
        ltoe_idx = amass.joint_names.index('ltoe')  # XXX
        ltoey_o = body_o.Jtr.detach().numpy()[0, ltoe_idx, 1]
        ltoey = body.Jtr.detach().numpy()[0, ltoe_idx, 1]
        zoff = ltoey - ltoey_o
        trans[:, 2] += zoff

    for frame in range(pose_body.shape[0]):
        pose_body_frame = pose_body[frame]
        root_orient_frame = root_orient[frame]
        root_trans_frame = trans[frame]
        pose_data = []
        for j in range(num_joints):
            if j == 0:
                T = fmc.Rp2T(fmc.A2R(root_orient_frame), root_trans_frame)
            else:
                T = fmc.R2T(
                    fmc.A2R(pose_body_frame[(j - 1) * 3 : (j - 1) * 3 + 3])
                )
            pose_data.append(T)
        motion.add_one_frame(pose_data)

    if fps is not None:
        resample(motion, fps)

    return motion


def estimate_mjshape(bm: BodyModel):
    '''
    Estimate SMPL shape parameters that match the MuJoCo robot.
    '''
    import nevergrad as ng

    JOINTS_POSE_EXCLUDE = {
        'root',
        'lowerback',
        'upperback',
        'chest',
        'lowerneck',
        'upperneck',
        'lclavicle',
        'rclavicle',
    }

    def basepose(bm, betas):
        body = bm(
            pose_body=pose_body_zeros,
            root_orient=th.zeros(1, 3),
            trans=th.zeros(1, 3),
            betas=th.from_numpy(betas).view(1, -1).to(th.float32),
        )
        return body.Jtr.detach().numpy()[0, 0:num_joints]

    joints_considered = []
    logs = 'Considering joints for shape estimation: '
    for i, jn in enumerate(amass.joint_names):
        if jn not in JOINTS_POSE_EXCLUDE:
            logs += jn + ' '
            sys.stdout.write(jn + ' ')
            joints_considered.append(i)
    log.info(logs)

    env = AMASSReplayEnv(robot='HumanoidAMASSPC')
    env.reset()
    p = env.p
    p.data.qpos[:] = 0
    p.data.qpos[3] = 1
    p.data.qvel[:] = 0
    p.forward()
    mj_pos = []
    for jn in amass.joint_names:
        mj_pos.append(p.named.data.site_xpos[f'robot/j_{jn}'][[0, 1, 2]])
    refpose = np.stack(mj_pos) - mj_pos[0]

    def posediff(bm, betas, verbose=False):
        pose = basepose(bm, betas)
        pose = pose - pose[0]
        if verbose:
            for i in range(1, num_joints):
                log.info(amass.joint_names[i], pose[i], refpose[i])
            log.info(pose[1])
            log.info(refpose[1])
        return np.square(
            pose[joints_considered] - refpose[joints_considered]
        ).sum()

    num_joints = len(amass.joint_names)
    pose_body_zeros = th.zeros((1, 3 * (num_joints - 1)))
    pose = basepose(bm, np.zeros(NUM_BETAS))

    optimizer = ng.optimizers.NGOpt(
        parametrization=ng.p.Array(shape=(NUM_BETAS,)).set_bounds(-5, 5),
        budget=4000,
        num_workers=4,
    )
    with futures.ThreadPoolExecutor(
        max_workers=optimizer.num_workers
    ) as executor:
        recom = optimizer.minimize(
            lambda val: posediff(bm, val), executor=executor, batch_mode=False
        )
    log.info(f'Opt {repr(recom.value)}')
    log.info(f'Loss {posediff(bm, recom.value, True)}')
    log.info(f'Base {posediff(bm, recom.value * 0, False)}')
    return recom.value


def corpus_clip_name(clip):
    corpus = osp.basename(osp.dirname(osp.dirname(clip)))
    corpus = {
        'BioMotionLab_NTroje': 'BMLrub',
        'DFaust_67': 'DFaust67',
        'Eyes_Japan_Dataset': 'EyesJapan',
        'MPI_HDM05': 'MPIHDM05',
        'MPI_Limits': 'MPILimits',
        'MPI_mosh': 'MPIMoSh',
        'SSM_synced': 'SSM',
        'TCD_handMocap': 'TCDHands',
        'Transitions_mocap': 'Transitions',
    }.get(corpus, corpus)

    cname = osp.basename(clip).removesuffix('_poses.npz')
    if corpus == 'CMU' or corpus.startswith('CMU_'):
        subject = osp.basename(osp.dirname(clip))
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
    elif corpus in (
        'SSM',
        'BMLhandball',
        'EKUT',
        'HumanEva',
        'KIT',
        'MPILimits',
        'MPIMoSh',
        'TotalCapture',
    ):
        subject = osp.basename(osp.dirname(clip))
        cname = cname.replace(' ', '-')
        cname = f'{subject}_{cname}'
    elif corpus == 'BMLrub':
        subject = osp.basename(osp.dirname(clip))
        cname = cname.split('_')[0]
        cname = f'{subject}_{cname}'
    elif corpus == 'MPIHDM05':
        subject = osp.basename(osp.dirname(clip))
        cname = cname.removesuffix('_120').removeprefix('HDM_')
    elif corpus == 'SSM':
        subject = osp.basename(osp.dirname(clip)).split('_')[1]
        cname = f'{subject}_{cname}'
    elif corpus == 'EyesJapan':
        subject = osp.basename(osp.dirname(clip))
        cname = '-'.join(cname.split('-')[0:2])
        cname = f'{subject}_{cname}'
    elif corpus == 'DanceDB':
        subject = osp.basename(osp.dirname(clip)).split('_')[1]
        cname = f'{subject}_{cname.removesuffix("_C3D")}'
    elif corpus == 'ACCAD':  # yes, this is a huge mess
        pdir = osp.basename(osp.dirname(clip))
        if pdir.startswith('Male'):
            subject = pdir[:5]
        elif pdir.startswith('Female'):
            subject = pdir[:7]
        elif pdir == 'MartialArtsWalksTurns_c3d':
            subject = 'X'
        else:
            subject = pdir
        if cname.startswith('B21 s2'):
            cname = 'B21-2'
        elif cname.startswith('B21 s3'):
            cname = 'B21-3'
        elif cname.startswith('C21 s2'):
            cname = 'C21-2'
        elif cname == 'General A8 -  Crourch to Lie (forward)':
            cname = 'A8-2'
        elif cname == 'G3 - Sidekick leading right':
            cname = 'G3-2'
        elif cname == 'male2 Subject Cal 2':
            cname = 'male2-2'
        elif cname == 'D9 - t2 - male2 blanks':
            cname = 'D9-2'
        elif cname == 'D9 - warm up to ready to warm up':
            cname = 'D9-3'
        elif cname.startswith('General A9 -   Lie (forward)'):
            if cname.endswith('t2'):
                cname = 'A9-3'
            else:
                cname = 'A9-2'
        elif (
            cname.startswith('General ')
            or cname.startswith('Run ')
            or cname.startswith('Walk ')
        ):
            if cname.endswith('t2') or cname.endswith(' a'):
                cname = cname.split()[1].rstrip('-') + '-2'
            elif cname.endswith('t3'):
                cname = cname.split()[1].rstrip('-') + '-3'
            else:
                cname = cname.split()[1].rstrip('-')
        elif cname.startswith('Extended '):
            cname = cname.replace(' ', '-')
        elif cname.endswith('t2') or cname.endswith(' a'):
            cname = cname.split()[0].rstrip('-') + '-2'
        elif cname.endswith('t3'):
            cname = cname.split()[0].rstrip('-') + '-3'
        else:
            cname = cname.split()[0].rstrip('-')
        cname = cname[0].upper() + cname[1:]
        cname = f'{subject}_{cname}'

    return corpus, cname
