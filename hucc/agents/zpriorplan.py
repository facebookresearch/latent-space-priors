# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import pickle
from copy import copy
from os import path as osp
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import fairmotion
import gym
import numpy as np
import torch as th
from dm_control.utils import rewards
from fairmotion.data import amass
from fairmotion.ops import motion as mops
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

import hucc
from hucc import mocap as hm
from hucc.agents import Agent
from hucc.mocap.datasets import th_R62R
from hucc.models import make_model
from hucc.spaces import box_space

log = logging.getLogger(__name__)

from torch import distributions as D
from torch import nn
from torch.nn import functional as F


def unstack_out(recon, in_keys=None):
    feat_shape = {
        'relxypos': 3,
        'rlvel': 3,
        'r6': 132,
        'rrelr6': 6,
        'br6': 132 - 6,
        'rquatr': 4,
        'bquat': 21 * 4,
    }
    in_keys = in_keys or ['relxypos', 'r6']
    r = {}
    off = 0
    for k in in_keys:
        n = feat_shape[k]
        try:
            r[k] = recon[:, off : off + n].cpu().numpy()
        except:
            r[k] = recon[:, off : off + n]
        off += n
    return r


def prior_sample(
    model,
    n_samples=1,
    length=None,
    z=None,
    ctx=None,
    y=None,
    temp=1,
    mtemp=None,
):
    if length is None:
        length = model.sample_length
    if ctx is None:
        ctx = model.sample_length - 1
    ctx = min(ctx, model.sample_length - 1)
    all_zs = None
    z_ctx = z
    y_idx = 0
    while True:
        if all_zs is None:
            gen_length = min(length, model.sample_length)
        else:
            gen_length = min(
                length - (all_zs.shape[2] - ctx), model.sample_length
            )
        if y is None or y.shape[1] == 1:
            zs = model.sample(
                n_samples=n_samples,
                chunk_size=1,
                z=z_ctx,
                y=y,
                temp=temp,
                mixture_temp=temp,
                sample_tokens=gen_length,
            ).permute(0, 2, 1)
        else:
            zs = model.sample(
                n_samples=n_samples,
                chunk_size=1,
                z=z_ctx,
                y=y[:, y_idx : y_idx + gen_length],
                temp=temp,
                mixture_temp=mtemp,
                sample_tokens=gen_length,
            ).permute(0, 2, 1)
            y_idx += gen_length - ctx
        if all_zs is None:
            all_zs = zs
        else:
            all_zs = th.cat([all_zs, zs[:, :, z_ctx.shape[2] :]], dim=-1)
        z_ctx = all_zs[:, :, -ctx:]
        all_zs = all_zs[:, :, :length]
        if all_zs.shape[2] >= length:
            break
    return all_zs


def vae_encode_motion(model, motion, mode='relxypos', in_keys=None):
    feats = hm.motion_to_jposrot(motion)

    cfg = model.cfg if hasattr(model, 'cfg') else DictConfig({})
    if in_keys is None:
        in_keys = cfg.dataset.inputs
    if hasattr(model, 'repr'):
        model = model.repr
    elif hasattr(model, 'vae'):
        if in_keys == ['zs']:
            in_keys = ['relxypos', 'r6']
        model = model.vae
    downs_t, strides_t = [0], [1]
    mdevice = next(model.parameters()).device
    if mode == 'relxypos' or mode == 'rlvel':
        inp = th.cat([th.from_numpy(feats[k]) for k in in_keys], dim=-1)
        if strides_t[-1] > 2:
            pad = strides_t[-1] ** sum(downs_t)  # Hotfix
        else:
            pad = 2 ** sum(downs_t)
        npad = pad - (inp.shape[0] % pad)
        if npad > 0:
            inp = th.cat([inp, th.zeros(npad, inp.shape[1])], dim=0)
        with th.no_grad():
            _, extra = model(inp.to(mdevice).unsqueeze(0))
        n = feats['rpos'].shape[0]
        out = extra['out'][0][:n]
        zs = extra['z'][0].T[:n]
    else:
        raise NotImplementedError()

    rfeats = unstack_out(out, in_keys)
    return (
        hm.jposrot_to_motion(
            rfeats, motion.skel, initial_pose=motion.get_pose_by_frame(0)
        ),
        zs,
    )  # , rfeats


class ZPriorPlanAgent(Agent):
    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        optim: SimpleNamespace,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        if not hasattr(model, 'pi'):
            raise ValueError('Model needs "pi" module')
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'ZPriorAgent requires a dict observation space (but got {type(env.observation_space)})'
            )
        # Lazy
        assert 'targets' in env.observation_space.spaces

        self._model = model
        self._replan_interval = int(cfg.replan_interval)
        self._history = int(cfg.history)
        self._context = int(cfg.context)
        self._horizon = int(cfg.horizon)
        assert self._replan_interval <= self._horizon
        self._rollouts = int(cfg.rollouts)
        self._score_fn = cfg.score_fn
        self._plan_act_offset = int(cfg.plan_act_offset)
        self._save_plans = bool(cfg.save_plans)
        self._supply_deltas = bool(cfg.supply_deltas)
        self._reencode_latents = bool(cfg.reencode_latents)
        self._reward_sigmoid = cfg.reward_sigmoid
        self._temp = float(cfg.prior.temp)
        self._mixture_temp = float(cfg.prior.mixture_temp)
        self._vae_without_tanh = bool(cfg.prior.vae_without_tanh)
        self._gamma = 1.0

        # Load prior
        if cfg.prior.model == 'from_path':
            prior_cfg = OmegaConf.load(
                osp.join(osp.dirname(cfg.prior.path), '.hydra/config.yaml')
            ).model
        else:
            prior_cfg = cfg.prior.model
        self._prior = make_model(
            prior_cfg,
            obs_space=box_space((prior_cfg.vae_input_dim,)),
            action_space=gym.spaces.Discrete(1),
        )
        mdevice = next(self._model.parameters()).device
        d = th.load(cfg.prior.path, map_location=mdevice)
        self._prior.load_state_dict(d.get('model', d))
        self._prior.eval()
        self._prior.to(mdevice)

        # Load AMASS skeleton
        root = hucc.__path__[0]
        bms = hm.load_amass_bms(osp.join(root, 'envs/assets/smplh'))
        mj_betas = np.load(osp.join(root, 'envs/assets/robot-smplh-shape.npz'))
        skel_betas = th.from_numpy(mj_betas['male']).to(th.float32).view(1, 16)
        self._skel = amass.create_skeleton_from_amass_bodymodel(
            bms['male'], skel_betas, len(amass.joint_names), amass.joint_names
        )

        self.set_checkpoint_attr('_model')

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        return gym.spaces.Dict(
            {
                'observation': env.observation_space.spaces['observation'],
                'z': box_space((32,)),  # XXX
            }
        )

    def init_plan(self, env):
        motion = fairmotion.core.motion.Motion(skel=self._skel, fps=1 / env.dt)
        pose = hm.qpos_to_pose(env.p, self._skel)
        # Start with a limited history
        motion.add_one_frame(pose.data)
        return SimpleNamespace(
            motion=motion, steps_since_replan=np.inf, zs=None, info=None
        )

    def step_plan(self, id, env, obs, plan):
        nd = env.p.named.data

        # Encode physically realized motion
        motion_hist = mops.cut(
            plan.motion, -self._history, plan.motion.num_frames()
        )
        z_hist = vae_encode_motion(self._prior, motion_hist, in_keys=['zs'])[1]

        # Sample new candidate trajectories by continuing the realized motion
        N = self._rollouts
        CTX = min(z_hist.shape[0], self._context)
        H = self._horizon
        z_hist = z_hist[-CTX:].unsqueeze(0).expand(N, CTX, z_hist.shape[1])
        zs = prior_sample(
            model=self._prior,
            z=z_hist.permute(0, 2, 1),
            n_samples=N,
            length=CTX + H,
            ctx=CTX,
            temp=self._temp,
            mtemp=self._mixture_temp,
        ).permute(0, 2, 1)
        # Decode candidates
        zs_with_hist = th.cat([z_hist, zs[:, -H:]], dim=1)
        if self._vae_without_tanh:
            eps = 1e-7
            zs_p = th.atanh(zs_with_hist.clamp(-1 + eps, 1 - eps))
        else:
            zs_p = zs_with_hist
        xs = self._prior.decode([zs_p.permute(0, 2, 1)])
        # Accumulate XY positions, get root rotations -- but only from the
        # current frame on
        xypos = th.cumsum(xs[:, -H:, :2], 1).cpu().numpy()
        rots = th_R62R(xs[:, -H:, 3:9]).cpu().numpy().reshape(-1, 3, 3)

        # Ok, se we want to score two criteria
        # 1) Whether we get closer to the target
        # 2) Whether we orient towards the target
        # Criterion 2) is included because we have so little data that we don't automatically get nice movements
        # all over the place...

        # Map observed (relative to current pos) target in local frame back to global (world) coords
        rel_target_local = obs['targets'].cpu()
        rel_target_global = R.from_matrix(
            nd.xmat['robot/torso'].reshape(3, 3)
        ).apply(rel_target_local)

        # For 1), diff relative target in world coords with position diffs
        # We project to X/Y since that's what we're interested in, but we still need 3D vectors
        # for later
        zero_last = np.zeros((xypos.shape[0], xypos.shape[1], 1))
        rel_traj_target_global = np.concatenate(
            [rel_target_global[:2].reshape(1, 1, -1) - xypos, zero_last],
            axis=-1,
        )
        dists = np.linalg.norm(rel_traj_target_global, axis=-1)

        # Project these relative (global) targets to the respective local frames, and
        # again project to X/Y
        # In terms of modeling in camera etc., X is forward, Z is up
        # For the character, we have Z as forward, Y as up.
        rel_traj_target_local = (
            R.from_matrix(rots).inv().apply(rel_target_global)
        )
        rel_traj_target_local = rel_traj_target_local[..., [0, 2]]

        # Compute angles between the relative local targets and the forward vector we'd have at that point
        rel_traj_target_local /= np.linalg.norm(
            rel_traj_target_local, axis=1
        ).reshape(-1, 1)
        cosines = np.inner(rel_traj_target_local, [[1, 0]])[:, 0]
        angles = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
        # Back to BxT
        angles = angles.reshape(-1, H)

        cur_dist = np.linalg.norm(rel_target_global)
        rel_target_cur = rel_target_local[[2, 0]]
        rel_target_cur /= np.linalg.norm(rel_target_cur)
        cur_angle = np.rad2deg(
            np.arccos(np.clip(np.inner(rel_target_cur, [1, 0]), -1.0, 1.0))
        )

        # Compute rewards for these criteria with dm_control's distance function.
        # In particular, we're interested in the highest improvements in terms of this reward
        # along the candidate trajectories.
        R_dist = rewards.tolerance(
            dists, (0, 0), margin=3, sigmoid=self._reward_sigmoid
        )
        R_ang = rewards.tolerance(
            angles, (0, 45), margin=90, sigmoid=self._reward_sigmoid
        )
        R_rdist = R_dist - rewards.tolerance(
            cur_dist, (0, 0), margin=3, sigmoid=self._reward_sigmoid
        )
        R_rang = R_ang - rewards.tolerance(
            cur_angle, (0, 45), margin=90, sigmoid=self._reward_sigmoid
        )
        besta = R_ang.max(axis=1).argmax()
        bestd = R_dist.max(axis=1).argmax()
        bestc = (R_rdist + R_rang).max(axis=1).argmax()

        if self._score_fn == 'distance':
            best = bestd
        elif self._score_fn == 'angle':
            best = besta
        elif self._score_fn == 'combined':
            best = bestc
        else:
            raise ValueError(f'No such score function: {self._score_fn}')

        log.debug(
            f'Plan step {id}/{int(obs["time"].item())} dist {cur_dist:.02f} -> {dists[best,-1].item():.02f}; ang {cur_angle:.01f} -> {angles[best,-1]:.01f}'
        )

        # Assemble plan for future acting
        robot_pos = env.robot_pos.copy()
        plan_info = {
            'robot_pos': robot_pos,
            'target': robot_pos + rel_target_global,
            'torso_pos': nd.xpos['robot/torso'],
            'xytrajs': xypos,
            'bestd': bestd,
            'besta': besta,
            'bestc': bestc,
            'best': best,
        }
        plan = copy(plan)
        plan.steps_since_replan = 0
        plan.info = plan_info

        if self._reencode_latents:
            zs_re = self._prior.encode(xs[best : best + 1], deterministic=True)[
                0
            ].permute(0, 2, 1)
            plan.zs = zs_re[0, CTX + self._plan_act_offset :].clone()
        else:
            plan.zs = zs[best, CTX + self._plan_act_offset :].clone()
        return plan

    @th.no_grad()
    def action(self, envs, obs) -> Tuple[th.Tensor, Any]:
        pdevice = next(self._prior.parameters()).device

        if not 'plans' in envs.ctx:
            envs.ctx['plans'] = [None for i in range(envs.num_envs)]
            envs.ctx['episodes'] = [-1 for i in range(envs.num_envs)]
        for i in th.where(obs['time'].view(-1).cpu() == 0)[0]:
            envs.ctx['plans'][i] = self.init_plan(envs.envs[i])
            envs.ctx['episodes'][i] += 1

        # For now let's plan separately in each env
        zs = []
        n_active = 0
        p_step = 0
        for i in range(envs.num_envs):
            # XXX save some time, we're evaluating one trial per env only
            step = int(obs['time'].view(-1)[i].item())
            if envs.ctx['episodes'][i] > 0:
                log.debug(f'Skip planning for env {i}')
                zs.append(th.zeros((32,), device=pdevice))
                continue
            n_active += 1
            p_step = max(p_step, step)

            env = envs.envs[i]
            plan = envs.ctx['plans'][i]
            pose = hm.qpos_to_pose(env.p, self._skel)
            plan.motion.add_one_frame(pose.data)

            if plan.steps_since_replan >= self._replan_interval:
                plan = self.step_plan(
                    i, env, {k: v[i] for k, v in obs.items()}, plan
                )
                envs.ctx['plans'][i] = plan

                if self._save_plans:
                    episode = envs.ctx['episodes'][i]
                    step = int(obs['time'].view(-1)[i].item())
                    th.save(plan, f'plan_{i:03d}_{episode:03d}_{step:04d}.pt')

            zs.append(plan.zs[plan.steps_since_replan])

            if self._supply_deltas:
                target_xy = (
                    plan.info['xytrajs'][plan.info['best']][
                        plan.steps_since_replan
                    ]
                    + plan.info['torso_pos'][:2]
                )
                env_xy = env.p.named.data.xpos['robot/torso'][:2]
                obs['observation'][i, -2:].copy_(
                    th.from_numpy(target_xy - env_xy)
                )

            plan.steps_since_replan += 1
        log.info(
            f'Step {p_step+1}: {envs.num_envs-n_active}/{envs.num_envs} envs done'
        )

        # Execute next action from plan
        z = th.stack(zs, dim=0)
        dist = self._model.pi({'observation': obs['observation'], 'z': z})
        action = dist.mean

        return action, None

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        pass
