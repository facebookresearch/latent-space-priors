# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
import time
from copy import deepcopy
from os import path as osp
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
import torch.distributions as D
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F

from hucc import ReplayBuffer, capture_graph
from hucc.agents import Agent
from hucc.models import make_model
from hucc.spaces import box_space
from hucc.utils import dim_select

log = logging.getLogger(__name__)


class GuidedSACAgent(Agent):
    '''
    Soft Actor-Critic agent with guidance from a model
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
                f'GuidedSACAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                f'GuidedSACAgent requires a dictionary observation space (but got {type(env.observation_space_space)})'
            )
        if not 'time' in env.observation_space.spaces:
            raise ValueError(f'GuidedSACAgent requires a "time" observation')
        if not 'observation' in env.observation_space.spaces:
            raise ValueError(
                f'GuidedSACAgent requires a "observation" observation'
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
        self._tanh_actions = bool(cfg.tanh_actions)
        self._normalize_actions = bool(cfg.get('normalize_actions', False))
        self._entropy_mean = bool(cfg.entropy_mean)
        self._action_cost = float(cfg.action_cost)
        self._action_cost_type = str(cfg.action_cost_type)
        assert self._action_cost_type in (
            'loss',
            'square',
            'delta_square',
            'square_and_delta_square',
        )

        # Optimize log(alpha) so that we'll always have a positive factor
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
        self._n_samples_since_update = 0
        self._cur_rewards: List[th.Tensor] = []

        self._target = deepcopy(model.hi)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._q = self._model.hi.q
        self._q_tgt = self._target.q
        self._lo_pi: nn.Module = self._model.lo.pi
        self._qup_g = None
        self._piup_g = None
        self._lo_g = None
        if cfg.graph:
            self._init_graph = True
        else:
            self._init_graph = False
        self._batch: Dict[str, th.Tensor] = {}
        self._bench = cfg.get('bench', False)
        self._t_last_update = -1.0

        self._action_factor = env.action_space.high[0]
        self._action_space = self.effective_action_space(env, cfg)['hi']
        self._cond_key = cfg.policy_lo.cond_key
        self._obs_lo = cfg.policy_lo.obs_key
        self._obs_space = env.observation_space
        self._obs_keys = list(self._obs_space.spaces.keys())
        self._obs_keys.remove('time')
        self._obs_keys.remove(self._obs_lo)
        if 'delta' in cfg.action_cost_type and cfg.action_cost > 0:
            self._obs_keys.append('prev_action')

        if self._entropy_mean:
            self._target_entropy = -1.0 * cfg.target_entropy_factor
        else:
            self._target_entropy = (
                -np.prod(self._action_space.shape) * cfg.target_entropy_factor
            )

        self._guide = lambda ns, x, n: x
        self._guide_c = 0
        self._guide_r = 0
        self._guide_c_max = 0
        self._guide_p = 0.0
        self._guide_p_anneal = 0
        self._guide_ctx = 0
        self._guide_exclude_samples = False
        self._guide_cond: Callable[[int], Optional[th.Tensor]] = lambda ns: None
        if (
            cfg.guide is not None
            and cfg.guide.path is not None
            and cfg.guide.path.endswith('.h5')
        ):
            log.info(f'Loading guiding frames from {cfg.guide.path}')
            import h5py
            import hdf5plugin

            with h5py.File(cfg.guide.path) as f:
                z_dists = f.get('train/zs', f.get('zs'))[:]
                zd_start = f.get('train/start', f.get('start'))[:]
                zd_end = np.concatenate([zd_start[1:], [z_dists.shape[0]]])
            # Mark time to go for easier sampling
            zd_remain = th.cat(
                [th.arange(e - s, 0, -1) for s, e in zip(zd_start, zd_end)]
            ).cuda()
            z_dists = th.from_numpy(z_dists).cuda()

            def sample_guide(z_dists, zd_remain, ns, x, n, temp):
                idx = th.multinomial((zd_remain >= n).float(), num_samples=ns)
                idxs = idx.tile(n).view(n, -1) + th.arange(
                    n, device=idx.device
                ).unsqueeze(1)
                dists = z_dists[idxs]
                mu, log_std = dists.chunk(2, -1)
                if temp <= 0:
                    smp = mu
                else:
                    smp = D.Normal(mu, F.softplus(log_std) * temp).sample()
                return th.tanh(smp.permute(1, 0, 2))

            self._guide = lambda ns, x, n: sample_guide(
                z_dists, zd_remain, ns, x, n, cfg.guide.temp
            )

            self._guide_c = cfg.guide.c
            try:
                self._guide_c_max = self._guide_m.sample_length
            except AttributeError:
                self._guide_c_max = 64
            self._guide_r = cfg.guide.r
            self._guide_p = cfg.guide.p
            self._guide_p_anneal = int(cfg.guide.p_anneal)
            self._guide_ctx = cfg.guide.ctx
            self._guide_exclude_samples = bool(cfg.guide.exclude_samples)
            assert self._guide_ctx == 0
        elif (
            cfg.guide is not None
            and cfg.guide.path is not None
            and cfg.guide.path.endswith('.pt')
        ):
            log.info(f'Loading guide from {cfg.guide.path}')
            if cfg.guide.model == 'from_path':
                guide_cfg = OmegaConf.load(
                    osp.join(osp.dirname(cfg.guide.path), '.hydra/config.yaml')
                ).model
            else:
                guide_cfg = cfg.guide.model
            self._guide_m = make_model(
                guide_cfg,
                obs_space=box_space((guide_cfg.vae_input_dim,)),
                action_space=gym.spaces.Discrete(cfg.guide.num_cond),
            )
            d = th.load(cfg.guide.path)
            self._guide_m.load_state_dict(d.get('model', d))
            self._guide_m.eval()
            self._guide_m.cuda()

            if cfg.guide.cond == 'rand':
                self._guide_cond_min = int(cfg.guide.min_cond)
                self._guide_cond_max = int(cfg.guide.num_cond)
                self._guide_cond = lambda ns: th.randint(
                    self._guide_cond_min,
                    self._guide_cond_max,
                    (ns, 1),
                    device='cuda',
                )
            elif isinstance(cfg.guide.cond, str):
                self._guide_cond_is = [
                    int(c) for c in str(cfg.guide.cond).split(':')
                ]
                self._guide_cond_v = (
                    th.tensor(self._guide_cond_is, device='cuda')
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                self._guide_cond = lambda ns: (
                    th.zeros(
                        (ns, 1, len(self._guide_cond_is)),
                        dtype=th.int64,
                        device='cuda',
                    )
                    + self._guide_cond_v
                )
            elif isinstance(cfg.guide.cond, Iterable):
                self._guide_cond_v = th.tensor(
                    [int(i) for i in cfg.guide.cond],
                    dtype=th.int64,
                    device='cuda',
                )
                self._guide_cond = lambda ns: self._guide_cond_v.index_select(
                    0,
                    th.randint(
                        0, self._guide_cond_v.shape[0], (ns,), device='cuda'
                    ),
                ).unsqueeze(1)
            elif cfg.guide.cond is not None:
                self._guide_cond_i = int(cfg.guide.cond)
                self._guide_cond = (
                    lambda ns: th.zeros((ns, 1), dtype=th.int64, device='cuda')
                    + self._guide_cond_i
                )
            # self._guide_cond = int(cfg.guide.cond) if cfg.guide.cond is not None else None
            self._guide = lambda ns, x, n: self._guide_m.sample(
                n_samples=ns,
                chunk_size=None,
                z=x,
                temp=cfg.guide.temp,
                sample_tokens=n,
                fp16=False,
                y=self._guide_cond(ns),
            )

            self._guide_c = cfg.guide.c
            try:
                self._guide_c_max = self._guide_m.sample_length
            except AttributeError:
                self._guide_c_max = 64
            self._guide_r = cfg.guide.r
            self._guide_p = cfg.guide.p
            self._guide_p_anneal = int(cfg.guide.p_anneal)
            self._guide_ctx = cfg.guide.ctx
            self._guide_exclude_samples = bool(cfg.guide.exclude_samples)
            assert self._guide_ctx >= 0
        elif cfg.guide is not None and cfg.guide.path is not None:
            raise ValueError(f'Can\'t handle guide path: {cfg.guide.path}')

        self._mod: Dict[str, nn.Module] = {}

        self.set_checkpoint_attr(
            '_model', '_target', '_optim', '_log_alpha', '_optim_alpha'
        )

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        obs_space_lo = env.observation_space.spaces[cfg.policy_lo.obs_key]

        spaces = {}
        if cfg.hide_obs_lo_from_hi:
            spaces['hi'] = gym.spaces.Dict(
                {
                    k: v
                    for k, v in env.observation_space.spaces.items()
                    if k != cfg.policy_lo.obs_key
                }
            )
        else:
            spaces['hi'] = deepcopy(env.observation_space)
        del spaces['hi'].spaces['time']
        spaces['hi'].spaces['unguided'] = box_space((1,))
        if 'delta' in cfg.action_cost_type and cfg.action_cost > 0:
            spaces['hi'].spaces['prev_action'] = box_space(
                (cfg.policy_lo.cond_dim,)
            )

        spaces['lo'] = gym.spaces.Dict(
            [
                ('observation', obs_space_lo),
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
        spaces['hi'] = box_space((n_actions_hi,), -1, 1)
        return spaces

    @Agent.training.setter
    def training(self, training) -> None:
        self._training = training
        if training:
            self._model.train()
        else:
            self._model.eval()
        self._model.lo.eval()

    def explore(self, env, obs) -> Tuple[th.Tensor, Any]:
        N = env.num_envs
        device = obs['observation'].device

        if self._n_samples < self._randexp_samples:
            action = th.stack(
                [th.from_numpy(self._action_space.sample()) for i in range(N)]
            ).to(device)
        else:
            with th.no_grad():
                dist = self._model.hi.pi(obs)
            action = dist.sample()

        if self._guide_c <= 0:
            return action, th.zeros(N, device=device, dtype=th.bool)
        if self._guide_p_anneal > 0 and self._n_samples >= self._guide_p_anneal:
            return action, th.zeros(N, device=device, dtype=th.bool)

        prev_actions = env.ctx.get('prev_actions', None)
        truncate_history = (obs['time'] == 0).long().view(-1)
        if truncate_history.all():
            prev_actions = None

        guide_steps_left = env.ctx.get(
            'guide_steps_left', th.zeros(N, device=device, dtype=th.int64)
        )
        guide_idx = env.ctx.get('guide_idx', th.zeros_like(guide_steps_left))
        curr_guide = env.ctx.get('guide', None)
        if self._n_samples < self._randexp_samples:
            # During initial random exploration, sample from guide only
            sample_guide = guide_steps_left == 0
        else:
            p = self._guide_p
            if self._guide_p_anneal > 0:
                p = self._guide_p * (1 - self._n_samples / self._guide_p_anneal)
            sample_guide = th.logical_and(
                guide_steps_left == 0, th.rand(N, device=device) < p
            )
        sample_guide |= truncate_history.bool()

        if sample_guide.any():
            # Sample new guidance actions
            if self._guide_r > 0:
                new_guide_steps = (
                    (
                        th.poisson(
                            th.zeros_like(guide_steps_left, dtype=th.float)
                            + self._guide_r
                        )
                        + 1
                    )
                    .clamp(max=self._guide_c)
                    .long()
                )
            else:
                new_guide_steps = (
                    th.zeros_like(guide_steps_left) + self._guide_c
                )
            if prev_actions is not None:
                new_guide_steps += prev_actions.shape[-1]
            new_guide_steps.clamp_(max=self._guide_c_max)

            if prev_actions == None or (obs['time'] >= self._guide_ctx).all():
                if prev_actions is None:
                    log.debug(
                        f'Sample from guide {new_guide_steps.max().item()} steps without ctx'
                    )
                else:
                    log.debug(
                        f'Sample from guide {new_guide_steps.max().item()} steps with ctx of {prev_actions.shape[-1]}'
                    )
                new_guide = self._guide(
                    N, prev_actions, new_guide_steps.max().item()
                )
                if prev_actions != None:
                    new_guide = new_guide[:, prev_actions.shape[-1] :]
                    new_guide_steps -= prev_actions.shape[-1]
            else:
                log.debug(
                    f'Sample from guide {new_guide_steps.max().item()} steps with ctx of {prev_actions.shape[-1]}'
                )
                new_guide = self._guide(
                    N, prev_actions, new_guide_steps.max().item()
                )
                for i in range(0, prev_actions.shape[-1]):
                    idx = th.where(obs['time'].view(-1) == i)[0]
                    if idx.numel() == 0:
                        continue
                    prev_actions_t = prev_actions[:, :, -i:] if i > 0 else None
                    missing = prev_actions.shape[-1] - i
                    new_guide_t = self._guide(
                        N,
                        prev_actions_t,
                        new_guide_steps.max().item() - missing,
                    )
                    new_guide[idx, missing:].copy_(new_guide_t[idx])
                new_guide = new_guide[:, prev_actions.shape[-1] :]
                new_guide_steps -= prev_actions.shape[-1]

            if curr_guide is None:
                curr_guide = th.zeros(
                    (N, self._guide_c, new_guide.shape[-1]),
                    device=new_guide.device,
                    dtype=new_guide.dtype,
                )

            curr_guide[:, : new_guide.shape[1]].masked_scatter_(
                sample_guide.view(-1, 1, 1),
                new_guide[th.where(sample_guide)[0]],
            )

            guide_steps_left = (
                new_guide_steps * sample_guide
                + guide_steps_left * ~sample_guide
            )
            guide_idx = guide_idx * ~sample_guide
        elif curr_guide is None:
            return action, th.zeros(N, device=device, dtype=th.bool)

        follow_guide = guide_steps_left > 0
        idx = th.where(follow_guide)[0]
        action[idx] = dim_select(curr_guide[idx], 1, guide_idx[idx])

        env.ctx['guide'] = curr_guide
        env.ctx['guide_steps_left'] = th.clamp(guide_steps_left - 1, min=0)
        env.ctx['guide_idx'] = th.clamp(guide_idx + 1, max=self._guide_c)
        if self._guide_ctx == 0:
            pass
        elif self._guide_ctx == 1 or prev_actions is None:
            env.ctx['prev_actions'] = action.unsqueeze(-1)
        elif self._guide_ctx > 1:
            env.ctx['prev_actions'] = th.cat(
                [
                    prev_actions[:, :, -self._guide_ctx + 1 :],
                    action.unsqueeze(-1),
                ],
                dim=-1,
            )

        return action, follow_guide

    @th.no_grad()
    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        N = env.num_envs
        device = obs['observation'].device
        prev_action = env.ctx.get('prev_action', None)
        if prev_action is None:
            prev_action = th.zeros(
                (N, self._action_space.shape[0]), device=device
            )

        mobs = dict(**obs)
        mobs['unguided'] = th.ones_like(obs['time'])
        mobs['prev_action'] = prev_action

        if not self.training:
            # Follow policy in eval mode
            dist = self._model.hi.pi(mobs)
            action = dist.mean
            guided = None
        else:
            action, guided = self.explore(env, mobs)

        orig_action = action
        if not self._tanh_actions:
            eps = 1e-7
            action = th.atanh(action.clamp(-1 + eps, 1 - eps))
        if self._normalize_actions:
            action = F.normalize(action, dim=-1)

        obs_lo = {}
        obs_lo['observation'] = obs[self._obs_lo]
        obs_lo[self._cond_key] = action
        dist_lo = self._lo_pi(obs_lo)
        if isinstance(dist_lo, th.Tensor):
            action_lo = dist_lo
        else:
            action_lo = dist_lo.mean

        env.ctx['prev_action'] = orig_action

        extra = {
            'action': orig_action,
            'guided': guided,
            'prev_action': prev_action,
        }
        if self.training:
            extra['viz'] = ['guided']
        return action_lo, extra

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        if self._t_last_update < 0:
            self._t_last_update = time.perf_counter()

        next_obs, reward, terminated, truncated, info = result
        action, guided, prev_action = (
            extra['action'],
            extra['guided'],
            extra['prev_action'],
        )
        # Ignore terminal state for truncated episodes
        done = terminated

        if self._action_cost > 0:
            if self._action_cost_type == 'square':
                reward = reward - self._action_cost * action.square().mean(
                    dim=-1
                ).view_as(reward)
            elif self._action_cost_type == 'delta_square':
                reward = reward - self._action_cost * (
                    action - prev_action
                ).square().mean(dim=-1).view_as(reward)
            elif self._action_cost_type == 'square_and_delta_square':
                reward = reward - self._action_cost * (
                    action.square().mean(dim=-1)
                    + (action - prev_action).square().mean(dim=-1)
                ).view_as(reward)

        d = dict(action=action, reward=reward, not_done=~done, unguided=~guided)
        for k in self._obs_keys:
            if k == 'prev_action':
                d['obs_prev_action'] = prev_action
                d['next_obs_prev_action'] = action
            else:
                d[f'obs_{k}'] = obs[k]
                d[f'next_obs_{k}'] = next_obs[k]

        self._buffer.put_row(d)
        self._cur_rewards.append(reward)

        self._n_steps += 1
        self._n_samples += done.nelement()
        self._n_samples_since_update += done.nelement()
        if self._buffer.size < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            if self._n_updates > 5 and self._bench:
                measures = []
                for i in range(10):
                    t = time.perf_counter()
                    self.update()
                    delta = time.perf_counter() - t
                    measures.append(delta * 1000)
                    log.info(f'Update in {int(1000*delta)}ms')
                log.info(
                    f'Update times: mean {np.mean(measures):.01f}ms median {np.median(measures):.01f}ms min {np.min(measures):.01f}ms max {np.max(measures):.01f}ms'
                )
                sys.exit(0)

            self.update()
            self._cur_rewards.clear()
            self._n_samples_since_update = 0

    def _update(self):
        t_start_update = time.perf_counter()
        mdevice = next(self._model.parameters()).device
        model = self._model.hi
        optim = self._optim.hi

        def act_logp(obs):
            dist = model.pi(obs)
            action = dist.rsample()
            if self._entropy_mean:
                log_prob = dist.log_prob(action).mean(dim=-1)
            else:
                log_prob = dist.log_prob(action).sum(dim=-1)
            action = action * self._action_factor
            if self._normalize_actions:
                action = F.normalize(action, dim=-1)
            return action, log_prob

        def q_loss_bwd(obs, obs_p, reward, not_done, batch_action):
            optim.q.zero_grad()

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)
                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(self._q_tgt(q_in), dim=-1).values
                backup = reward + self._gamma * not_done * (
                    q_tgt - self._log_alpha.detach().exp() * log_prob_p
                )

            # Q-Function update
            q_in = dict(action=batch_action, **obs)
            q = self._q(q_in)
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
            q = th.min(self._q(q_in), dim=-1).values
            if self._action_cost_type == 'loss':
                pi_loss = (
                    self._log_alpha.detach().exp() * log_prob
                    + self._action_cost * a.square().mean(dim=-1)
                    - q
                ).mean()
            else:
                pi_loss = (self._log_alpha.detach().exp() * log_prob - q).mean()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.pi.parameters(), self._clip_grad_norm
                )
            return pi_loss, log_prob.mean()

        # if self._guide_c > 0:
        #    unguided = th.where(self._buffer._b['unguided'] == True)[0]
        for _ in range(self._num_updates):
            # if self._guide_c > 0:
            #    self._batch = self._buffer.get_batch_where(
            #        self._bsz, indices=unguided, device=mdevice, out=self._batch
            #    )
            # else:
            self._batch = self._buffer.get_batch(
                self._bsz, device=mdevice, out=self._batch
            )
            batch = self._batch
            reward = batch['reward']
            not_done = batch['not_done']
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            obs['unguided'] = batch['unguided']
            if not 'unguided_dummy' in batch:
                batch['unguided_dummy'] = th.ones_like(batch['unguided'])
            obs_p['unguided'] = batch['unguided_dummy']
            if not self._guide_exclude_samples:
                obs['unguided'].fill_(1)

            # Q-function update
            if self._init_graph:
                log.info('Building graph for q-function update')
                self._qup_g, self._qup_out = capture_graph(
                    lambda: q_loss_bwd(
                        obs, obs_p, reward, not_done, batch['action']
                    )
                )

            if self._qup_g is not None:
                self._qup_g.replay()
                q_loss = self._qup_out
            else:
                q_loss = q_loss_bwd(
                    obs, obs_p, reward, not_done, batch['action']
                )
            optim.q.step()

            # Policy update
            obs['unguided'].fill_(1)
            for param in model.q.parameters():
                param.requires_grad_(False)

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
                pi_loss, log_prob = pi_loss_pwd(obs)
            optim.pi.step()

            for param in model.q.parameters():
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

            with th.no_grad():
                for tp, p in zip(
                    self._target.q.parameters(), model.q.parameters()
                ):
                    tp.data.lerp_(p.data, 1.0 - self._polyak)

        # These are the stats for the last update
        self.tbw_add_scalar('Loss/Policy', pi_loss.item())
        self.tbw_add_scalar('Loss/QValue', q_loss.item())
        self.tbw_add_scalar('Health/Entropy', -log_prob.item())
        if self._optim_alpha:
            self.tbw_add_scalar('Health/Alpha', self._log_alpha.exp().item())
        if self._n_updates % 100 == 1:
            self.tbw.add_scalars(
                'Health/GradNorms',
                {
                    k: v.grad.norm().item()
                    for k, v in model.named_parameters()
                    if v.grad is not None
                },
                self.n_samples,
            )
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
