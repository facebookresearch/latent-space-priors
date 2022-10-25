# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig
from torch import distributions as D
from torch import nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function

import hucc.models.blocks
from hucc import ReplayBuffer, capture_graph
from hucc.agents import Agent
from hucc.spaces import th_flatten

log = logging.getLogger(__name__)


class SACAgent(Agent):
    '''
    Soft Actor-Critic agent. See config/sac.yaml for defaults.
    '''

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
        if not hasattr(model, 'q'):
            raise ValueError('Model needs "q" module')
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f'SACAgent requires a continuous (Box) action space (but got {type(env.action_space)})'
            )

        self._model = model
        self._optim = optim
        self._bsz = int(cfg.batch_size)
        self._gamma = float(cfg.gamma)
        self._polyak = float(cfg.polyak)
        self._rpbuf_size = int(cfg.rpbuf_size)
        self._samples_per_update = int(cfg.samples_per_update)
        self._num_updates = int(cfg.num_updates)
        self._policy_delay = int(cfg.policy_delay)
        self._policy_target = str(cfg.policy_target)
        self._warmup_samples = int(cfg.warmup_samples)
        self._randexp_samples = int(cfg.randexp_samples)
        self._flatten_obs = bool(cfg.flatten_obs)
        self._clip_grad_norm = float(cfg.clip_grad_norm)

        self._target_entropy = (
            -np.prod(env.action_space.shape) * cfg.target_entropy_factor
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

        self._target = deepcopy(model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._q = self._model.q
        self._q_tgt = self._target.q
        self._qup_g = None
        self._piup_g = None
        if cfg.graph:
            self._init_graph = True
        else:
            self._init_graph = False
        self._batch: Dict[str, th.Tensor] = {}
        self._bench = cfg.get('bench', False)

        self._action_space = env.action_space
        self._action_factor = env.action_space.high[0]
        self._obs_space = env.observation_space
        if isinstance(self._obs_space, gym.spaces.Dict):
            self._obs_keys = list(self._obs_space.spaces.keys())
        else:
            self._obs_keys = []

        self.set_checkpoint_attr(
            '_model', '_target', '_optim', '_log_alpha', '_optim_alpha'
        )

    @th.no_grad()
    def action(self, env, obs) -> Tuple[th.Tensor, Any]:
        if self._n_samples < self._randexp_samples and self.training:
            action = th.stack(
                [
                    th.from_numpy(self._action_space.sample())
                    for i in range(env.num_envs)
                ]
            ).to(list(self._model.parameters())[0].device)
        else:
            mobs = (
                th_flatten(self._obs_space, obs) if self._flatten_obs else obs
            )
            dist = self._model.pi(mobs)
            assert (
                dist.has_rsample
            ), f'rsample() required for policy distribution'
            if self.training:
                action = dist.sample() * self._action_factor
            else:
                action = dist.mean * self._action_factor
        return action, None

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        next_obs, reward, terminated, truncated, info = result
        d = dict(
            action=action, reward=reward, not_done=th.logical_not(terminated)
        )
        if self._obs_keys:
            for k in self._obs_keys:
                d[f'obs_{k}'] = obs[k]
                d[f'next_obs_{k}'] = next_obs[k]
        else:
            d['obs'] = obs
            d['next_obs'] = next_obs

        self._buffer.put_row(d)
        self._cur_rewards.append(reward)

        self._n_steps += 1
        self._n_samples += terminated.nelement()
        self._n_samples_since_update += terminated.nelement()
        if self._buffer.size < self._warmup_samples:
            return
        if self._n_samples_since_update >= self._samples_per_update:
            do_profile = False
            if self._n_updates >= (0 if do_profile else 5) and self._bench:
                import time

                measures = []
                if not do_profile:
                    for i in range(10):
                        th.cuda.synchronize()
                        t = time.time()
                        self.update()
                        d = time.time() - t
                        measures.append(d * 1000)
                        log.info(f'Update in {int(1000*d)}ms')
                    log.info(
                        f'Update times: mean {np.mean(measures):.01f}ms median {np.median(measures):.01f}ms min {np.min(measures):.01f}ms max {np.max(measures):.01f}ms'
                    )
                else:
                    th.cuda.synchronize()
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    ) as prof:
                        for i in range(2):
                            t = time.time()
                            self.update()
                            d = time.time() - t
                            measures.append(d * 1000)
                            log.info(f'Update in {int(1000*d)}ms')
                    log.info(
                        f'Update times: mean {np.mean(measures):.01f}ms median {np.median(measures):.01f}ms min {np.min(measures):.01f}ms max {np.max(measures):.01f}ms'
                    )
                    print(
                        prof.key_averages().table(
                            sort_by="cuda_time_total", row_limit=10
                        )
                    )
                    print(
                        prof.key_averages().table(
                            sort_by="self_cpu_memory_usage", row_limit=10
                        )
                    )
                    prof.export_chrome_trace("/tmp/trace.json")
                    # prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")

                import sys

                sys.exit(0)

            self.update()
            self._cur_rewards.clear()
            self._n_samples_since_update = 0

    def _update(self):
        mdevice = next(self._model.parameters()).device

        def act_logp(obs):
            dist = self._model.pi(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = action * self._action_factor
            return action, log_prob

        def q_loss_bwd(obs, obs_p, reward, not_done, batch_action):
            self._optim.q.zero_grad()

            # Backup for Q-Function
            with th.no_grad():
                a_p, log_prob_p = act_logp(obs_p)
                if self._flatten_obs:
                    q_in = th.cat([obs_p, a_p], dim=1)
                else:
                    q_in = dict(action=a_p, **obs_p)
                q_tgt = th.min(self._q_tgt(q_in), dim=-1).values
                backup = reward + self._gamma * not_done * (
                    q_tgt - self._log_alpha.detach().exp() * log_prob_p
                )

            # Q-Function update
            if self._flatten_obs:
                q_in = th.cat([obs, batch_action], dim=1)
            else:
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
                    self._model.q.parameters(), self._clip_grad_norm
                )
            return q_loss

        def pi_loss_bwd(obs):
            self._optim.pi.zero_grad()

            a, log_prob = act_logp(obs)
            if self._flatten_obs:
                q_in = th.cat([obs, a], dim=1)
            else:
                q_in = dict(action=a, **obs)
            if self._policy_target == 'min':
                q = th.min(self._q(q_in), dim=-1).values
            elif self._policy_target == 'mean':
                q = th.mean(self._q(q_in), dim=-1)
            else:
                raise ValueError(
                    f'Invalid policy target: {self._policy_target}'
                )
            pi_loss = (self._log_alpha.detach().exp() * log_prob - q).mean()
            pi_loss.backward()
            if self._clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self._model.pi.parameters(), self._clip_grad_norm
                )
            return pi_loss, log_prob.mean()

        for i in range(self._num_updates):
            self._batch = self._buffer.get_batch(
                self._bsz, device=mdevice, out=self._batch
            )
            batch = self._batch
            reward = batch['reward']
            not_done = batch['not_done']
            if self._obs_keys:
                obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
                obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
                if self._flatten_obs:
                    batch['flat_obs'] = th_flatten(
                        self._obs_space, obs, batch.get('flat_obs', None)
                    )
                    batch['flat_next_obs'] = th_flatten(
                        self._obs_space, obs_p, batch.get('flat_next_obs', None)
                    )
                    obs = batch['flat_obs']
                    obs_p = batch['flat_next_obs']
            else:
                obs = batch['obs']
                obs_p = batch['next_obs']

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
            self._optim.q.step()

            if i % self._policy_delay == 0:
                # Policy update
                for param in self._model.q.parameters():
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
                    pi_loss, log_prob = pi_loss_bwd(obs)
                self._optim.pi.step()

                for param in self._model.q.parameters():
                    param.requires_grad_(True)

                # Optional temperature update
                if self._optim_alpha:
                    alpha_loss = -(
                        self._log_alpha.exp()
                        * (log_prob.detach() + self._target_entropy)
                    )
                    self._optim_alpha.zero_grad()
                    alpha_loss.backward()
                    self._optim_alpha.step()

            # Update target network
            with th.no_grad():
                for tp, p in zip(
                    self._target.q.parameters(), self._model.q.parameters()
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
                    for k, v in self._model.named_parameters()
                    if v.grad is not None
                },
                self.n_samples,
            )

        print_log = True
        if self._samples_per_update < 10:
            print_log = self._n_updates % 100 == 0
        if not print_log:
            return
        avg_cr = th.cat(self._cur_rewards).mean().item()
        log.info(
            f'Sample {self._n_samples}, up {self._n_updates*self._num_updates}, avg cur reward {avg_cr:+0.3f}, pi loss {pi_loss.item():+.03f}, q loss {q_loss.item():+.03f}, entropy {-log_prob.item():+.03f}, alpha {self._log_alpha.exp().item():.03f}'
        )
