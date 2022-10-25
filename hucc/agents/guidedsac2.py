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
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import distributions as D
from torch import nn
from torch.nn import functional as F

from hucc import ReplayBuffer, capture_graph
from hucc.agents import Agent
from hucc.models import make_model
from hucc.spaces import box_space
from hucc.utils import dim_select

log = logging.getLogger(__name__)


class GuidedSAC2Agent(Agent):
    '''
    Soft Actor-Critic KL-regularized towards a guide
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
        self._entropy_mean = bool(cfg.entropy_mean)
        self._guide_kl_in_backup = bool(cfg.guide_kl_in_backup)
        self._reverse_kl = bool(cfg.reverse_kl)
        self._mixture_norm_kl = bool(cfg.mixture_norm_kl)
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
        self._obs_keys.remove(self._obs_lo)
        if 'delta' in cfg.action_cost_type and cfg.action_cost > 0:
            self._obs_keys.append('prev_action')

        if self._entropy_mean:
            self._target_entropy = -1.0 * cfg.target_entropy_factor
        else:
            self._target_entropy = (
                -np.prod(self._action_space.shape) * cfg.target_entropy_factor
            )

        self._guide = lambda x, n: x
        self._guide_ctx = 0
        self._init_guide_alpha = 0.0
        self._guide_alpha_anneal = 0
        self._guide_cond: Callable[[int], Optional[th.Tensor]] = lambda ns: None
        if cfg.guide is not None and cfg.guide.path is not None:
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

            if isinstance(cfg.guide.cond, str):
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
            elif cfg.guide.cond is not None:
                self._guide_cond_i = int(cfg.guide.cond)
                self._guide_cond = (
                    lambda ns: th.zeros((ns, 1), dtype=th.int64, device='cuda')
                    + self._guide_cond_i
                )

            # log.info(f'Converting guide to fp16')
            # from hucc.models.prior import _convert_conv_weights_to_fp16
            # self._guide_m.apply(_convert_conv_weights_to_fp16)
            self._guide_m.eval()
            self._guide_m.cuda()
            self._guide = lambda x, n: self._guide_m.sample(
                n_samples=n,
                chunk_size=None,
                z=x,
                temp=0,
                sample_tokens=1 if x is None else x.shape[-1] + 1,
                fp16=False,
                get_dists=True,
                y=self._guide_cond(n),
            )[1]
            self._guide_ctx = cfg.guide.ctx
            self._init_guide_alpha = float(cfg.guide.alpha)
            self._guide_alpha = th.tensor(self._init_guide_alpha, device='cuda')
            self._guide_alpha_anneal = int(cfg.guide.alpha_anneal)

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
        mobs['prev_action'] = prev_action

        if not self.training:
            # Follow policy in eval mode
            dist = self._model.hi.pi(mobs)
            action = dist.mean
        else:
            if self._n_samples < self._randexp_samples:
                action = th.stack(
                    [
                        th.from_numpy(self._action_space.sample())
                        for i in range(N)
                    ]
                ).to(device)
            else:
                dist = self._model.hi.pi(mobs)
                action = dist.sample()

        orig_action = action
        if not self._tanh_actions:
            eps = 1e-7
            action = th.atanh(action.clamp(-1 + eps, 1 - eps))

        obs_lo = {}
        obs_lo['observation'] = obs[self._obs_lo]
        obs_lo[self._cond_key] = action
        with th.no_grad():
            dist_lo = self._lo_pi(obs_lo)
        action_lo = dist_lo.mean

        env.ctx['prev_action'] = orig_action
        return action_lo, (orig_action, prev_action)

    def _fill_guide(self):
        I = self._buffer.interleave
        N = self._buffer.max
        B = self._buffer._b
        idx = th.where(B['guide_missing'] == True)[0]
        time = B['obs_time'][idx].clamp(max=self._guide_ctx).long()
        actions = B['action']
        tdevice = idx.device
        bsz = 1000

        # TODO in most cases, the distribution for the next action can be
        # just... taken from the next action.
        for i in range(int(time.min().item()), self._guide_ctx + 1):
            tidxx = idx[th.where(time == i)[0]]
            if tidxx.numel() == 0:
                continue
            for j in range(0, tidxx.shape[0], bsz):
                tidx = tidxx[j : j + bsz]
                if i == 0:
                    g_dist = self._guide(None, tidx.shape[0])[-1]
                    prev_actions = actions[tidx].unsqueeze(-1)
                    g_dist_p = self._guide(prev_actions.cuda(), tidx.shape[0])[
                        -1
                    ]
                else:
                    action_idx = tidx.unsqueeze(-1) + th.arange(
                        -i * I, -1, I, device=tdevice
                    )
                    prev_actions = actions[action_idx % N].permute(
                        0, 2, 1
                    )  # BxCxT
                    g_dist = self._guide(prev_actions.cuda(), tidx.shape[0])[-1]
                    prev_actions = th.cat(
                        [
                            prev_actions[:, :, -self._guide_ctx + 1 :],
                            actions[tidx].unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    g_dist_p = self._guide(prev_actions.cuda(), tidx.shape[0])[
                        -1
                    ]

                if isinstance(g_dist, D.MixtureSameFamily):
                    mdist = g_dist.mixture_distribution
                    cdist = g_dist.component_distribution.base_dist
                    mdist_p = g_dist_p.mixture_distribution
                    cdist_p = g_dist_p.component_distribution.base_dist
                    B['guide_dist_mix'].index_copy_(
                        0, tidx, mdist.logits.squeeze(1).to(tdevice)
                    )
                    B['guide_dist_mu'].index_copy_(
                        0, tidx, cdist.loc.squeeze(1).to(tdevice)
                    )
                    B['guide_dist_std'].index_copy_(
                        0, tidx, cdist.scale.squeeze(1).to(tdevice)
                    )
                    B['next_guide_dist_mix'].index_copy_(
                        0, tidx, mdist_p.logits.squeeze(1).to(tdevice)
                    )
                    B['next_guide_dist_mu'].index_copy_(
                        0, tidx, cdist_p.loc.squeeze(1).to(tdevice)
                    )
                    B['next_guide_dist_std'].index_copy_(
                        0, tidx, cdist_p.scale.squeeze(1).to(tdevice)
                    )
                else:
                    B['guide_dist_mu'].index_copy_(
                        0, tidx, g_dist.loc.to(tdevice)
                    )
                    B['guide_dist_std'].index_copy_(
                        0, tidx, g_dist.scale.to(tdevice)
                    )
                    B['next_guide_dist_mu'].index_copy_(
                        0, tidx, g_dist_p.loc.to(tdevice)
                    )
                    B['next_guide_dist_std'].index_copy_(
                        0, tidx, g_dist_p.scale.to(tdevice)
                    )

        B['guide_missing'].index_fill_(0, idx, False)

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
        action, prev_action = extra
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

        try:
            mixture_size: int = self._guide_m.prior.mixture_size
        except AttributeError:
            mixture_size = self._guide_m.mixture_size
        if mixture_size > 1:
            assert self._entropy_mean == False
            guide_dummy = th.zeros(
                (action.shape[0], mixture_size, action.shape[1]),
                device=action.device,
            )
        else:
            guide_dummy = th.zeros_like(action)
        d = dict(
            action=action,
            reward=reward,
            not_done=~done,
            guide_dist_mu=guide_dummy,
            guide_dist_std=guide_dummy,
            next_guide_dist_mu=guide_dummy,
            next_guide_dist_std=guide_dummy,
            guide_missing=th.ones_like(done),
        )
        if mixture_size > 1:
            d['guide_dist_mix'] = th.zeros(
                (action.shape[0], mixture_size), device=action.device
            )
            d['next_guide_dist_mix'] = th.zeros(
                (action.shape[0], mixture_size), device=action.device
            )
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
            self._fill_guide()

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
        for p in self._model.parameters():
            mdevice = p.device
            break
        model = self._model.hi
        optim = self._optim.hi

        if self._guide_alpha_anneal > 0:
            ga = self._init_guide_alpha * (
                1 - self._n_samples / self._guide_alpha_anneal
            )
            self._guide_alpha.fill_(max(0, min(1, ga)))

        def act_logp(obs):
            dist = model.pi(obs)
            action = dist.rsample()
            if self._entropy_mean:
                log_prob = dist.log_prob(action).mean(dim=-1)
            else:
                log_prob = dist.log_prob(action).sum(dim=-1)
            action = action * self._action_factor
            return action, log_prob, dist

        def q_loss_bwd(
            obs,
            obs_p,
            reward,
            not_done,
            batch_action,
            guide_dist_p,
            guide_p_sample,
        ):
            optim.q.zero_grad()

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p, dist = act_logp(obs_p)
                q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(self._q_tgt(q_in), dim=-1).values
                if self._guide_kl_in_backup:
                    if isinstance(guide_dist, D.Normal):
                        if self._reverse_kl:
                            guide_kl = D.kl_divergence(
                                dist.base_dist, guide_dist_p
                            )
                        else:
                            guide_kl = D.kl_divergence(
                                guide_dist_p, dist.base_dist
                            )
                    else:
                        if self._reverse_kl:
                            # Take a new sample b/c we want to get the KL before tanh
                            guide_kl = -guide_dist_p.log_prob(
                                dist.base_dist.rsample()
                            )
                        else:
                            if self._mixture_norm_kl:
                                guide_kl = -(
                                    dist.base_dist.log_prob(guide_p_sample).sum(
                                        -1
                                    )
                                    - guide_dist_p.log_prob(guide_p_sample)
                                )
                            else:
                                guide_kl = -dist.base_dist.log_prob(
                                    guide_p_sample
                                ).sum(-1)

                    backup = reward + self._gamma * not_done * (
                        q_tgt
                        - self._log_alpha.detach().exp() * log_prob_p
                        - self._guide_alpha * guide_kl.mean(dim=-1)
                    )
                else:
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

        def pi_loss_bwd(obs, guide_dist, guide_sample):
            optim.pi.zero_grad()

            a, log_prob, dist = act_logp(obs)
            q_in = dict(action=a, **obs)
            q = th.min(self._q(q_in), dim=-1).values
            if isinstance(guide_dist, D.Normal):
                if self._reverse_kl:
                    guide_kl = D.kl_divergence(dist.base_dist, guide_dist).mean(
                        dim=-1
                    )
                else:
                    guide_kl = D.kl_divergence(guide_dist, dist.base_dist).mean(
                        dim=-1
                    )
            else:
                if self._reverse_kl:
                    # Take a new sample b/c we want to get the KL before tanh
                    guide_kl = -guide_dist.log_prob(dist.base_dist.rsample())
                else:
                    if self._mixture_norm_kl:
                        guide_kl = -(
                            dist.base_dist.log_prob(guide_sample).sum(-1)
                            - guide_dist.log_prob(guide_sample)
                        )
                    else:
                        guide_kl = -dist.base_dist.log_prob(guide_sample).sum(
                            -1
                        )
            if self._action_cost_type == 'loss':
                pi_loss = (
                    self._log_alpha.detach().exp() * log_prob
                    + self._guide_alpha * guide_kl
                    + self._action_cost * a.square().mean(dim=-1)
                    - q
                ).mean()
            else:
                pi_loss = (
                    self._log_alpha.detach().exp() * log_prob
                    + self._guide_alpha * guide_kl
                    - q
                ).mean()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.pi.parameters(), self._clip_grad_norm
                )
            return (pi_loss, log_prob.mean(), guide_kl.detach().mean())

        for _ in range(self._num_updates):
            self._batch = self._buffer.get_batch(
                self._bsz, device=mdevice, out=self._batch
            )
            batch = self._batch
            reward = batch['reward']
            not_done = batch['not_done']
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            if 'guide_dist_mix' in batch:
                guide_dist = D.MixtureSameFamily(
                    D.Categorical(logits=batch['guide_dist_mix']),
                    D.Independent(
                        D.Normal(
                            batch['guide_dist_mu'],
                            batch['guide_dist_std'],
                            validate_args=False,
                        ),
                        1,
                    ),
                    validate_args=False,
                )
                guide_dist_p = D.MixtureSameFamily(
                    D.Categorical(logits=batch['next_guide_dist_mix']),
                    D.Independent(
                        D.Normal(
                            batch['next_guide_dist_mu'],
                            batch['next_guide_dist_std'],
                            validate_args=False,
                        ),
                        1,
                    ),
                    validate_args=False,
                )

                guide_sample = guide_dist.sample()
                if '_guide_sample' in batch:
                    batch['_guide_sample'].copy_(guide_sample)
                else:
                    batch['_guide_sample'] = guide_sample
                guide_sample = batch['_guide_sample']
                guide_p_sample = guide_dist_p.sample()
                if '_guide_p_sample' in batch:
                    batch['_guide_p_sample'].copy_(guide_p_sample)
                else:
                    batch['_guide_p_sample'] = guide_p_sample
                guide_p_sample = batch['_guide_p_sample']
            else:
                guide_dist = D.Normal(
                    batch['guide_dist_mu'],
                    batch['guide_dist_std'],
                    validate_args=False,
                )
                guide_dist_p = D.Normal(
                    batch['next_guide_dist_mu'],
                    batch['next_guide_dist_std'],
                    validate_args=False,
                )
                guide_sample = None
                guide_p_sample = None

            # Q-function update
            if self._init_graph:
                log.info('Building graph for q-function update')
                self._qup_g, self._qup_out = capture_graph(
                    lambda: q_loss_bwd(
                        obs,
                        obs_p,
                        reward,
                        not_done,
                        batch['action'],
                        guide_dist_p,
                        guide_p_sample,
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
                    batch['action'],
                    guide_dist_p,
                    guide_p_sample,
                )
            optim.q.step()

            # Policy update
            for param in model.q.parameters():
                param.requires_grad_(False)

            if self._init_graph:
                log.info('Building graph for policy update')
                self._piup_g, self._piup_out = capture_graph(
                    lambda: pi_loss_bwd(obs, guide_dist, guide_sample),
                    pool=self._qup_g.pool(),
                )
                self._init_graph = False

            if self._piup_g is not None:
                self._piup_g.replay()
                pi_loss, log_prob, guide_kl = self._piup_out
            else:
                pi_loss, log_prob, guide_kl = pi_loss_bwd(
                    obs, guide_dist, guide_sample
                )
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
        self.tbw_add_scalar('Health/GuideKL', guide_kl.item())
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
        if self._n_updates % hist_freq == 1:
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
            f'Sample {self._n_samples}, up {self._n_updates*self._num_updates}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.item():+.03f}, prior_kl {guide_kl.item():+.03f}, alpha {self._log_alpha.exp().item():.03f} in {elapsed}/{elapsed_up}ms'
        )
        self._t_last_update = time.perf_counter()
