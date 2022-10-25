# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import sys
import time
from copy import copy, deepcopy
from os import path as osp
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F

import hucc
from hucc import ReplayBuffer, capture_graph
from hucc.agents import Agent
from hucc.models import make_model
from hucc.spaces import box_space
from hucc.utils import dim_select

log = logging.getLogger(__name__)


def _parse_list(s, dtype):
    if s is None:
        return []
    return list(map(dtype, str(s).split('#')))


class SACHRLAgent(Agent):
    '''
    A generic HRL SAC agent, in which the high-level action space is continuous.
    It includes the optimization when dealing with temporal abstraction proposed
    in "Dynamics-Aware Embeddings".
    This agent doesn't provide a means to actually train the low-level agent; it
    is assumed to be fixed and will be loaded from a checkpoint instead.
    '''

    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        optim: SimpleNamespace,
        cfg: DictConfig,
    ):
        super().__init__(cfg)

        if not hasattr(model, 'hi'):
            raise ValueError('Model needs "hi" module')
        if not hasattr(model, 'lo'):
            raise ValueError('Model needs "lo" module')
        if not hasattr(model.hi, 'pi'):
            raise ValueError('Model needs "hi.pi" module')
        if not hasattr(model.hi, 'q'):
            raise ValueError('Model needs "hi.q" module')
        if not hasattr(model.lo, 'pi'):
            raise ValueError('Model needs "lo.pi" module')
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f'SACHRLAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'SACHRLAgent requires a dictionary observation space (but got {type(env.observation_space_space)})'
            )
        if not 'time' in env.observation_space.spaces:
            raise ValueError(f'SACHRLAgent requires a "time" observation')
        if not 'observation' in env.observation_space.spaces:
            raise ValueError(
                f'SACHRLAgent requires a "observation" observation'
            )

        self._model = model
        self._optim = optim
        self._bsz = int(cfg.batch_size)
        self._gamma = float(cfg.gamma)
        self._polyak = float(cfg.polyak)
        self._rpbuf_size = int(cfg.rpbuf_size)
        self._samples_per_update = int(cfg.samples_per_update)
        self._num_updates = int(cfg.num_updates)
        self._warmup_samples = int(cfg.warmup_samples)
        self._randexp_samples = int(cfg.randexp_samples)
        self._clip_grad_norm = float(cfg.clip_grad_norm)
        self._dyne_updates = bool(cfg.dyne_updates)
        self._action_interval = int(cfg.action_interval)
        self._action_interval_start = self._action_interval
        self._action_interval_anneal = int(cfg.action_interval_anneal)
        self._action_interval_min = int(cfg.action_interval_min)
        self._action_factor_hi = float(cfg.action_factor_hi)
        self._tanh_actions = bool(cfg.tanh_actions)
        self._action_cost = float(cfg.action_cost)
        self._action_cost_type = str(cfg.action_cost_type)
        assert self._action_cost_type in (
            'square',
            'delta_square',
            'square_and_delta_square',
        )
        n_actions_hi = int(cfg.policy_lo.cond_dim)

        self._upsampler: Any = None
        self._upsampler_m: Any = None
        self._interpolate_actions = False
        if cfg.upsampler is not None and cfg.upsampler.type is not None:
            if cfg.upsampler.type == 'interpolate':
                self._interpolate_actions = True
            elif cfg.upsampler.type == 'repeat':
                self._upsampler = lambda x, n: x.unsqueeze(1).expand(
                    x.shape[0], n, x.shape[1]
                )
            elif cfg.upsampler.type == 'matrix':
                log.info(f'Loading upsampler from {cfg.upsampler.path}')
                mat = th.load(cfg.upsampler.path)
                self._upsampler = nn.Linear(
                    mat.shape[0], mat.shape[1], bias=False
                )
                self._upsampler.weight.detach().copy_(mat.T)
            elif cfg.upsampler.type == 'model':
                # ???
                n_out = n_actions_hi
                self._upsampler_m = make_model(
                    cfg.upsampler.model,
                    obs_space=box_space((n_out,)),
                    action_space=None,
                )
                log.info(f'Loading upsampler from {cfg.upsampler.path}')
                d = th.load(cfg.upsampler.path)
                self._upsampler_m.load_state_dict(d.get('model', d))
                self._upsampler_m.eval()
                self._upsampler = lambda x, n: self._upsampler_m.decode(x)
            elif cfg.upsampler.type == 'jukebox':
                up_cfg = cfg.upsampler.model
                if up_cfg == 'from_path':
                    up_cfg = OmegaConf.load(
                        osp.dirname(cfg.upsampler.path) + '/.hydra/config.yaml'
                    )
                    up_cfg = up_cfg.model
                self._upsampler_m = make_model(
                    up_cfg,
                    obs_space=box_space((135,)),  # XXX [relxypos,r6]
                    action_space=None,
                )
                log.info(f'Loading upsampler from {cfg.upsampler.path}')
                d = th.load(cfg.upsampler.path)
                self._upsampler_m.load_state_dict(d.get('model', d))
                # log.info(f'Converting upsampler to fp16')
                # from hucc.models.prior import _convert_conv_weights_to_fp16
                # self._upsampler_m.apply(_convert_conv_weights_to_fp16)
                self._upsampler_m.eval()
                self._upsampler_m.cuda()
                if self._upsampler_m.x_cond:
                    self._upsampler = lambda x, n: self._upsampler_m.sample(
                        n_samples=x.shape[0],
                        chunk_size=None,
                        z_conds=[x.unsqueeze(-1)],
                        temp=0,
                        sample_tokens=n,
                        fp16=False,
                    )
                else:
                    self._upsampler = lambda x, n: self._upsampler_m.sample(
                        n_samples=x.shape[0],
                        chunk_size=None,
                        z=x.unsqueeze(-1),
                        temp=0,
                        sample_tokens=n,
                        fp16=False,
                    )
        self._record_upsampled_actions = bool(cfg.record_upsampled_actions)

        self._qup_g = None
        self._piup_g = None
        if cfg.graph:
            self._init_graph = True
        else:
            self._init_graph = False
        self._batch: Dict[str, th.Tensor] = {}
        self._bench = cfg.get('bench', False)
        self._t_last_update = -1.0

        hide_from_lo = _parse_list(cfg.hide_from_lo, int)
        self._action_hi_key = cfg.policy_lo.cond_key
        self._obs_lo = cfg.policy_lo.obs_key
        obs_space = env.observation_space.spaces[cfg.policy_lo.obs_key]
        assert len(obs_space.shape) == 1
        self._features_lo = list(range(obs_space.shape[0]))
        for f in hide_from_lo:
            self._features_lo.remove(f)

        self._target_entropy_factor = cfg.target_entropy_factor
        self._target_entropy = (
            -np.prod(n_actions_hi) * cfg.target_entropy_factor
        )
        log_alpha = np.log(cfg.alpha)
        if cfg.optim_alpha is None:
            self._log_alpha = th.tensor(log_alpha)
            self._optim_alpha = None
        else:
            if cfg.graph:
                self._log_alpha = th.tensor(
                    log_alpha, device='cuda', requires_grad=True
                )
            else:
                self._log_alpha = th.tensor(log_alpha, requires_grad=True)
            self._optim_alpha = hydra.utils.instantiate(
                cfg.optim_alpha, [self._log_alpha]
            )

        rpbuf_device = cfg.rpbuf_device if cfg.rpbuf_device != 'auto' else None
        self._buffer = ReplayBuffer(
            size=self._rpbuf_size, interleave=env.num_envs, device=rpbuf_device
        )
        self._staging = ReplayBuffer(
            size=self._action_interval * env.num_envs,
            interleave=env.num_envs,
            device=rpbuf_device,
        )
        self._n_samples_since_update = 0
        self._cur_rewards: List[th.Tensor] = []
        self._d_batchin = None

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._action_space_lo = env.action_space
        self._action_factor_lo = env.action_space.high[0]
        as_hi = self.effective_action_space(env, cfg)['hi']
        self._action_space_hi = as_hi
        self._obs_space = self.effective_observation_space(env, cfg)['hi']['q']
        self._obs_keys = list(self._obs_space.spaces.keys())
        self._obs_keys.append('time')

        self.set_checkpoint_attr(
            '_model', '_target', '_optim', '_log_alpha', '_optim_alpha'
        )

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        hide_from_lo = _parse_list(cfg.hide_from_lo, int)
        obs_space = env.observation_space.spaces[cfg.policy_lo.obs_key]
        assert len(obs_space.shape) == 1
        features_lo = list(range(obs_space.shape[0]))
        for f in hide_from_lo:
            features_lo.remove(f)

        spaces = {}
        if cfg.hide_obs_lo_from_hi:
            spaces_hi = gym.spaces.Dict(
                {
                    k: v
                    for k, v in env.observation_space.spaces.items()
                    if k != cfg.policy_lo.obs_key
                }
            )
        else:
            spaces_hi = deepcopy(env.observation_space)
        del spaces_hi.spaces['time']
        if 'delta' in cfg.action_cost_type and cfg.action_cost > 0:
            spaces_hi.spaces['prev_action'] = box_space(
                (cfg.policy_lo.cond_dim,)
            )
        spaces['hi'] = {'q': deepcopy(spaces_hi), 'pi': deepcopy(spaces_hi)}
        if cfg.dyne_updates:
            spaces['hi']['q'].spaces['time_in_action'] = gym.spaces.Discrete(
                cfg.action_interval
            )
            spaces['hi']['q'].spaces['action_length'] = gym.spaces.Discrete(
                cfg.action_interval
            )

        lo_min = env.observation_space[cfg.policy_lo.obs_key].low[features_lo]
        lo_max = env.observation_space[cfg.policy_lo.obs_key].high[features_lo]
        # XXX The order of the observation spaces is important since e.g. with
        # DIAYN, we train the model on cat([observation,condition])
        spaces['lo'] = gym.spaces.Dict(
            [
                ('observation', gym.spaces.Box(lo_min, lo_max)),
                (
                    cfg.policy_lo.cond_key,
                    box_space((1, cfg.policy_lo.cond_dim)),
                ),
            ]
        )
        return spaces

    @staticmethod
    def effective_action_space(env: gym.Env, cfg: DictConfig):
        n_actions_hi = int(cfg.policy_lo.cond_dim)
        spaces = {}
        spaces['lo'] = env.action_space
        if cfg.tanh_actions:
            spaces['hi'] = box_space((n_actions_hi,), -1, 1)
        else:
            spaces['hi'] = box_space((n_actions_hi,))
        return spaces

    @Agent.training.setter
    def training(self, training) -> None:
        self._training = training
        if training:
            self._model.train()
        else:
            self._model.eval()
        self._model.lo.eval()

    def action_hi(
        self, env, obs
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        N = env.num_envs
        device = obs['observation'].device
        time = obs['time'].long().view(-1)

        interval = self._action_interval
        if not self.training or self._action_interval_anneal <= 0:
            step = time.remainder(interval).long()
            new_length = th.zeros_like(step) + interval
            last_action_time = None
            steps_left = None
        else:
            steps_left = env.ctx.get('steps_left', None)
            last_action_time = env.ctx.get('last_action_time', None)
            if last_action_time is None:
                step = time
                assert step.sum() == 0
            else:
                step = time - last_action_time
                step *= th.logical_and(steps_left > 0, time > 0)
            new_length = (
                th.poisson(th.zeros(N, device=device) + interval)
                .clamp(self._action_interval_min, self._action_interval_start)
                .long()
            )
        keep_action = step != 0

        action = env.ctx.get('action_hi', None)
        prev_action = env.ctx.get('action_hi_prev', None)
        if action is None and keep_action.any().item():
            raise RuntimeError('Need to take first action at time=0')
        if action is None or not keep_action.all().item():
            if self._n_samples < self._randexp_samples and self.training:
                new_action = th.stack(
                    [
                        th.from_numpy(self._action_space_hi.sample())
                        for i in range(env.num_envs)
                    ]
                ).to(list(self._model.parameters())[0].device)
            else:
                # obs_wo_time = copy(obs)
                # obs_wo_time['time_in_action'] = th.zeros_like(obs['time'])
                dist = self._model.hi.pi(obs)
                assert (
                    dist.has_rsample
                ), f'rsample() required for hi-level policy distribution'
                if self.training:
                    new_action = dist.sample()
                else:
                    new_action = dist.mean
            if action is None:
                action = new_action
                prev_action = new_action.clone()
            else:
                m = keep_action.unsqueeze(1)
                prev_action = m * prev_action + th.logical_not(m) * action
                action = m * action + th.logical_not(m) * new_action
            if last_action_time is None:
                env.ctx['last_action_time'] = time
                env.ctx['steps_left'] = new_length
            else:
                m = keep_action
                env.ctx['last_action_time'] = (
                    m * last_action_time + th.logical_not(m) * time
                )
                env.ctx['steps_left'] = (
                    m * steps_left + th.logical_not(m) * new_length
                )

        env.ctx['action_hi'] = action
        env.ctx['action_hi_prev'] = prev_action
        env.ctx['steps_left'] = env.ctx['steps_left'] - 1

        if self._interpolate_actions:
            alpha = (step.float() / self._action_interval).unsqueeze(1)
            action = alpha * action + (1 - alpha) * prev_action

        action_length = (
            time - env.ctx['last_action_time'] + env.ctx['steps_left'] + 1
        )
        return action, th.logical_not(keep_action), step, action_length

    @th.no_grad()
    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        if self._action_interval_anneal > 0:
            self._action_interval = max(
                self._action_interval_min,
                int(
                    np.ceil(
                        self._action_interval_start
                        * (1 - self._n_samples / self._action_interval_anneal)
                    )
                ),
            )

        N = env.num_envs
        device = obs['observation'].device
        prev_action = env.ctx.get('prev_action', None)
        if prev_action is None:
            prev_action = th.zeros(
                (N, self._action_space_hi.shape[0]), device=device
            )

        mobs = dict(**obs)
        mobs['prev_action'] = prev_action

        action_hi, _, step, action_length = self.action_hi(env, mobs)
        orig_action_hi = action_hi

        if not self._tanh_actions:
            eps = 1e-7
            action_hi = action_hi.clamp(-1 + eps, 1 - eps)
            action_hi = th.atanh(action_hi)
        action_hi = action_hi * self._action_factor_hi

        interval = self._action_interval
        if interval > 1 and self._upsampler is not None:
            if (step == 0).any():
                if isinstance(self._upsampler, nn.Module):
                    self._upsampler = self._upsampler.to(action_hi.device)
                elif self._upsampler_m is not None:
                    self._upsampler_m = self._upsampler_m.to(action_hi.device)
                max_length = action_length.max().item()
                if max_length > 1:
                    upsampled = self._upsampler(action_hi, max_length).view(
                        N, max_length, -1
                    )
                else:
                    upsampled = action_hi.view(N, 1, -1)
                if not 'upsampled' in env.ctx:
                    env.ctx['upsampled'] = th.zeros(
                        (N, self._action_interval_start, upsampled.shape[-1]),
                        device=upsampled.device,
                    )
                env.ctx['upsampled'][:, :max_length].copy_(upsampled)
                upsampled = env.ctx['upsampled']
            else:
                upsampled = env.ctx['upsampled']
            action_hi = dim_select(upsampled, 1, step)
            env.ctx['upsampled'] = upsampled
        elif (step > 0).any():
            # account for any actions left when we annealed to 1
            assert 'upsampled' in env.ctx
            upsampled = env.ctx['upsampled']
            upsampled[:, :interval].copy_(action_hi.unsqueeze(1))
            action_hi = dim_select(upsampled, 1, step)
            env.ctx['upsampled'] = upsampled

        obs_lo = {}
        obs_lo['observation'] = obs[self._obs_lo][:, self._features_lo]
        obs_lo[self._action_hi_key] = action_hi
        dist = self._model.lo.pi(obs_lo)
        try:
            action_lo = dist.mean * self._action_factor_lo
        except AttributeError:
            action_lo = dist['hi_action']  # XXX testing

        env.ctx['prev_action'] = orig_action_hi
        if action_hi.shape[-1] < 10:
            return action_lo, {
                'a_hi': orig_action_hi,
                'prev_action': prev_action,
                'step': step,
                'action_length': action_length,
                'viz': ['a_hi'],
            }
        else:
            if self._record_upsampled_actions:
                return action_lo, {'a_hi': action_hi}
            else:
                return action_lo, {
                    'a_hi': orig_action_hi,
                    'prev_action': prev_action,
                    'step': step,
                    'action_length': action_length,
                    'a_hi_ups': action_hi,
                    'viz': ['action_length'],
                }

    def step(
        self,
        env,
        obs,
        action,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        if self._t_last_update < 0:
            self._t_last_update = time.perf_counter()

        next_obs, reward, terminated, truncated, info = result
        # Ignore terminal state for truncated episodes
        done = terminated

        if self._action_cost > 0:
            act = extra['a_hi']
            prev_action = extra['prev_action']
            if self._action_cost_type == 'square':
                cost = act.square().mean(dim=-1).view_as(reward)
                reward = (
                    reward - (self._action_cost / self._action_interval) * cost
                )
            elif self._action_cost_type == 'delta_square':
                cost = (act - prev_action).square().mean(dim=-1).view_as(reward)
                reward = reward - self._action_cost * cost
            elif self._action_cost_type == 'square_and_delta_square':
                cost1 = act.square().mean(dim=-1).view_as(reward)
                cost2 = (
                    (act - prev_action).square().mean(dim=-1).view_as(reward)
                )
                reward = (
                    reward
                    - (self._action_cost / self._action_interval)
                    * (cost1 + cost2)
                    - self._action_cost * cost2
                )

        d = dict(
            reward=reward,
            terminal=done,
            step=extra['step'],
            action=extra['a_hi'],
            action_length=extra['action_length'],
        )
        for k in self._obs_keys:
            if k == 'prev_action':
                d['obs_prev_action'] = prev_action
                d['next_obs_prev_action'] = extra['a_hi']
            elif k == 'time_in_action':
                d[f'obs_time'] = obs['time']
                d[f'next_obs_time'] = next_obs['time']
            elif k == 'action_length':
                continue
            else:
                d[f'obs_{k}'] = obs[k]
                d[f'next_obs_{k}'] = next_obs[k]

        self._staging.put_row(d)
        self._cur_rewards.append(reward)

        if self._staging.size == self._staging.max:
            self._staging_to_buffer()

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()

        if self._action_interval_anneal > 0:
            self._action_interval = max(
                self._action_interval_min,
                int(
                    np.ceil(
                        self._action_interval_start
                        * (1 - self._n_samples / self._action_interval_anneal)
                    )
                ),
            )

        ilv = self._staging.interleave
        if self._buffer.size + self._staging.size - ilv < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            if self._n_updates > 5 and self._bench:
                measures = []
                for i in range(10):
                    t = time.time()
                    self.update()
                    delta = time.time() - t
                    measures.append(delta * 1000)
                    log.info(f'Update in {int(1000*delta)}ms')
                log.info(
                    f'Update times: mean {np.mean(measures):.01f}ms median {np.median(measures):.01f}ms min {np.min(measures):.01f}ms max {np.max(measures):.01f}ms'
                )
                sys.exit(0)

            self.update()
            self._cur_rewards.clear()
            self._n_samples_since_update = 0

    def _staging_to_buffer(self):
        ilv = self._staging.interleave
        buf = self._staging
        assert buf._b is not None
        batch: Dict[str, th.Tensor] = dict()
        idx = buf.start + th.arange(0, ilv, device=buf.device)
        c_max = buf._b['action_length'][idx].max().item()
        for k in set(buf._b.keys()):
            s = [
                buf._b[k].index_select(0, (idx + i * ilv) % buf.max)
                for i in range(c_max)
            ]
            batch[k] = th.stack(s, dim=1)

        # c = action_freq
        # i = batch['step']
        # Next action at c - i steps further, but we'll take next_obs so
        # access it at c - i - 1
        # import ipdb
        # ipdb.set_trace()
        c = batch['action_length'][:, 0]
        next_action = (c - 1) - batch['step'][:, 0]
        # If we have a terminal before, use this instead
        terminal = batch['terminal'].clone()
        for j in range(1, c_max):
            terminal[:, j] |= terminal[:, j - 1] * (j < c)
        first_terminal = c - terminal.sum(dim=1)
        # Lastly, the episode could have ended with a timeout, which we can
        # detect if we took another action (i == 0) prematurely. This will screw
        # up the reward summation, but hopefully it doesn't hurt too much.
        next_real_action = th.zeros_like(next_action) + c
        for j in range(1, c_max):
            idx = th.where(batch['step'][:, j] == 0)[0]
            next_real_action[idx] = next_real_action[idx].clamp(0, j - 1)
        next_idx = th.min(th.min(next_action, first_terminal), next_real_action)

        # Sum up discounted rewards until next c - i - 1
        reward = batch['reward'][:, 0].clone()
        for j in range(1, c_max):
            reward += self._gamma**j * batch['reward'][:, j] * (next_idx >= j)

        not_done = th.logical_not(dim_select(batch['terminal'], 1, next_idx))
        exclude = ['time_in_action', 'action_length']
        obs = {
            k: batch[f'obs_{k}'][:, 0]
            for k in self._obs_keys
            if k not in exclude
        }
        obs['time_in_action'] = batch['step'][:, 0:1].clone()
        obs['action_length'] = c - 1
        obs_p = {
            k: dim_select(batch[f'next_obs_{k}'], 1, next_idx)
            for k in self._obs_keys
            if k not in exclude
        }
        obs_p['time_in_action'] = th.zeros_like(obs['time_in_action'])
        # Assume we'd continue with the same interval
        obs_p['action_length'] = obs['action_length']

        gamma_exp = th.zeros_like(reward) + self._gamma
        gamma_exp.pow_(next_idx + 1)

        db = dict(
            reward=reward,
            not_done=not_done,
            gamma_exp=gamma_exp,
            action=batch['action'][:, 0],
        )
        for k, v in obs.items():
            db[f'obs_{k}'] = v
        for k, v in obs_p.items():
            db[f'next_obs_{k}'] = v

        self._buffer.put_row(db)

    def _update(self):
        t_start_update = time.perf_counter()
        mdevice = next(self._model.parameters()).device
        model = self._model.hi
        optim = self._optim.hi

        def act_logp(obs):
            dist = model.pi(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob

        def q_loss_bwd(obs, obs_p, reward, not_done, gamma_exp, batch_action):
            optim.q.zero_grad()

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)
                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(self._target.hi.q(q_in), dim=-1).values
                backup = reward + gamma_exp * not_done * (
                    q_tgt - self._log_alpha.detach().exp() * log_prob_p
                )

            # Q-Function update
            q_in = dict(action=batch_action, **obs)
            q = model.q(q_in)
            q1 = q[:, 0]
            q2 = q[:, 1]
            q1_loss = F.mse_loss(q1, backup)
            q2_loss = F.mse_loss(q2, backup)
            q_loss = q1_loss + q2_loss
            q_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.q.parameters(), self._clip_grad_norm
                )
            return q_loss

        def pi_loss_bwd(obs):
            optim.pi.zero_grad()

            a, log_prob = act_logp(obs)
            q_in = dict(action=a, **obs)
            q = th.min(model.q(q_in), dim=-1).values
            pi_loss = (self._log_alpha.detach().exp() * log_prob - q).mean()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.pi.parameters(), self._clip_grad_norm
                )
            return pi_loss, log_prob.mean()

        indices = None
        if not self._dyne_updates:
            assert (
                self._buffer.start == 0 or self._buffer.size == self._buffer.max
            )
            indices = th.where(
                self._buffer._b['obs_time'][: self._buffer.size] == 0
            )[0]

        if not '_zero_time' in self._batch:
            self._batch['_zero_time'] = th.zeros(
                self._bsz, device=mdevice, dtype=th.long
            )
            self._batch['_default_action_length'] = th.zeros(
                self._bsz, device=mdevice, dtype=th.long
            )
        self._batch['_default_action_length'].fill_(self._action_interval - 1)

        for _ in range(self._num_updates):
            if indices is not None:
                self._batch = self._buffer.get_batch_where(
                    self._bsz, device=mdevice, indices=indices, out=self._batch
                )
            else:
                self._batch = self._buffer.get_batch(
                    self._bsz, device=mdevice, out=self._batch
                )
            batch = self._batch
            reward = batch['reward']
            not_done = batch['not_done']
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}

            # Q-function update
            if self._init_graph:
                log.info('Building graph for q-function update')
                self._qup_g, self._qup_out = capture_graph(
                    lambda: q_loss_bwd(
                        obs,
                        obs_p,
                        reward,
                        not_done,
                        batch['gamma_exp'],
                        batch['action'],
                    )
                )

            if self._qup_g is not None:
                self._qup_g.replay()
                q_loss = self._qup_out
            else:
                q_loss = q_loss_bwd(
                    obs,
                    obs_p,
                    reward,
                    not_done,
                    batch['gamma_exp'],
                    batch['action'],
                )
            optim.q.step()

            # Policy update
            for param in model.q.parameters():
                param.requires_grad_(False)

            # No time input for policy, and Q-functions are queried as if step
            # would be 0 (i.e. we would take an action)
            obs['time_in_action'] = batch['_zero_time']
            obs['action_length'] = batch['_default_action_length']

            if self._init_graph:
                log.info('Building graph for policy update')
                self._piup_g, self._piup_out = capture_graph(
                    lambda: pi_loss_bwd(obs), pool=self._qup_g.pool()
                )
                self._init_graph = False

            if self._piup_g is not None:
                self._piup_g.replay()
                pi_loss, log_prob = self._piup_out
            else:
                pi_loss, log_prob = pi_loss_bwd(obs)
            optim.pi.step()

            for param in self._model.hi.q.parameters():
                param.requires_grad_(True)

            # Optional temperature update
            if self._optim_alpha:
                alpha_loss = -(
                    self._log_alpha.exp()
                    * (log_prob + self._target_entropy).detach()
                )
                self._optim_alpha.zero_grad()
                alpha_loss.backward()
                self._optim_alpha.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.hi.q.parameters(), model.q.parameters()
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        # These are the stats for the last update
        self.tbw_add_scalar('Loss/Policy', pi_loss.item())
        self.tbw_add_scalar('Loss/QValue', q_loss.item())
        self.tbw_add_scalar('Health/Entropy', -log_prob.item())
        if self._optim_alpha:
            self.tbw_add_scalar('Health/Alpha', np.exp(self._log_alpha.item()))
        if self._n_updates % 100 == 1:
            self.tbw.add_scalars(
                'Health/GradNorms',
                {
                    k: v.grad.norm().item()
                    for k, v in self._model.named_parameters()
                    if v.grad is not None
                },
                self.n_samples,
            )
        if self._action_interval_anneal > 0:
            self.tbw_add_scalar('Health/ActionInterval', self._action_interval)
        hist_freq = 100 if self.n_samples > 1e6 else 10
        if self._n_updates % 10 == 1:
            last_actions = self._buffer.get_batch_back(100, keys=['action'])[
                'action'
            ]
            self.tbw.add_histogram(
                'Health/Actions', last_actions, self.n_samples
            )

        avg_cr = th.cat(self._cur_rewards).mean().item()
        elapsed = int(1000 * (time.perf_counter() - self._t_last_update))
        elapsed_up = int(1000 * (time.perf_counter() - t_start_update))
        log.info(
            f'Sample {self._n_samples}, up {self._n_updates*self._num_updates}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.item():+.03f}, alpha {self._log_alpha.exp().item():.03f} in {elapsed}/{elapsed_up}ms'
        )
        self._t_last_update = time.perf_counter()
