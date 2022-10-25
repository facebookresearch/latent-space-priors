# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import math
import time
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import gym
import hydra
import numpy as np
import torch as th
import torch.distributed as thd
import torch.distributions as D
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hucc import ReplayBuffer, capture_graph
from hucc.agents import Agent
from hucc.agents.utils import discounted_bwd_cumsum_
from hucc.spaces import box_space, th_flatten

log = logging.getLogger(__name__)


class VMPOCoMicAgent(Agent):
    '''
    V-MPO agent for CoMic (Hasenclever et al., 2020)
    '''

    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        optim: SimpleNamespace,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        if not hasattr(model, 'encoder'):
            raise ValueError('Model needs "encoder" module')
        if not hasattr(model, 'pi'):
            raise ValueError('Model needs "pi" module')
        if not hasattr(model, 'v'):
            raise ValueError('Model needs "v" module')

        self._model = model
        self._optim = optim
        self._batch_num_traj = int(cfg.batch_num_traj)
        self._batch_traj_len = int(cfg.batch_traj_len)
        self._bsz = self._batch_num_traj * self._batch_traj_len  # in samples
        self._gamma = float(cfg.gamma)
        self._epochs = float(cfg.epochs)
        self._target_steps = int(cfg.target_steps)
        self._lg = th.tensor(
            [float(cfg.eta), float(cfg.alpha_mu), float(cfg.alpha_sigma)],
            requires_grad=True,
            device=list(model.parameters())[0].device,
        )
        # self._eta = self._lg[0]
        # self._alpha_mu = self._lg[1]
        # self._alpha_sigma = self._lg[2]
        self._eps_eta = float(cfg.eps_eta)
        self._eps_alpha_mu = float(cfg.eps_alpha_mu)
        self._eps_alpha_sigma = float(cfg.eps_alpha_sigma)
        self._kl_reg = float(cfg.kl_reg)
        self._prior = str(cfg.prior)
        self._ar1_alpha = float(cfg.ar1_alpha)
        self._max_steps = int(cfg.max_steps)
        self._latent_dim = int(cfg.latent_dim)
        self._tanh_latents = bool(cfg.tanh_latents)
        self._multi_reward = bool(cfg.multi_reward)
        self._encoder_interval = int(cfg.encoder_interval)
        self._encup_g = None
        self._piup_g = None
        self._vup_g = None
        self._fwd_g = None
        self._batch = None
        self._fwd_batch: Dict[str, th.Tensor] = {}
        self._init_graph = bool(cfg.graph)
        self._init_fwd_graph = self._init_graph

        self._optim_lg = hydra.utils.instantiate(cfg.optim_lg, [self._lg])

        self._rp_size = (
            max(self._bsz, math.ceil(self._batch_traj_len / env.num_envs))
            * self._target_steps
        )
        self._buffer = None
        self._buffers: List[ReplayBuffer] = []
        self._rpbuf_device = (
            cfg.rpbuf_device if cfg.rpbuf_device != 'auto' else None
        )
        self._popart = hasattr(self._model, 'popart')

        self._distributed = False
        if cfg.distributed.size > 1:
            thd.init_process_group(
                backend='nccl' if th.cuda.is_available() else 'gloo',
                rank=cfg.distributed.rank,
                world_size=cfg.distributed.size,
                init_method=f'file://{cfg.distributed.rdvu_path}',
            )
            self._distributed = True
            thd.barrier()
            for p in self._model.parameters():
                thd.broadcast(p.data, src=0)
        self._log_factor = cfg.distributed.size

        self._target_encoder = deepcopy(self._model.encoder)
        self._target_pi = deepcopy(self._model.pi)
        # We'll never need gradients for the target network
        for param in self._target_encoder.parameters():
            param.requires_grad_(False)
        for param in self._target_pi.parameters():
            param.requires_grad_(False)

        self._obs_space = env.observation_space
        if not isinstance(self._obs_space, gym.spaces.Dict):
            raise ValueError("Dictionary observation required")
        self._obs_keys = list(self._obs_space.spaces.keys())
        if 'time' in self._obs_keys:
            self._obs_keys.remove('time')

        self.set_checkpoint_attr(
            '_model',
            '_target_encoder',
            '_target_pi',
            '_optim',
            '_lg',
            '_optim_lg',
        )
        self._last_update_time = time.perf_counter()

    @property
    def supports_async_step(self):
        return True

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        orig = copy(env.observation_space.spaces)
        if 'time' in orig:
            del orig['time']
        if 'tick' in orig:
            del orig['tick']
        encoder_obs = {}
        if cfg.encoder_proprio_input:
            encoder_obs['observation'] = orig['observation']
        encoder_obs['reference'] = orig['reference']
        if cfg.encoder_prev_input:
            encoder_obs['z_prev'] = box_space((cfg.latent_dim,))
        return {
            'encoder': gym.spaces.Dict(encoder_obs),
            'pi': gym.spaces.Dict(
                {
                    'observation': orig['observation'],
                    'z': box_space((cfg.latent_dim,)),
                }
            ),
            'v': gym.spaces.Dict(orig),
        }

    @staticmethod
    def effective_action_space(env: gym.Env, cfg: DictConfig):
        return {
            'encoder': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(cfg.latent_dim,),
                dtype=np.float32,
            ),
            'pi': env.action_space,
            'v': env.action_space,
        }

    @th.no_grad()
    def action(self, env, obs) -> Tuple[th.Tensor, th.Tensor]:
        z_prev = env.ctx.get('z_prev', None)
        if z_prev is None:
            device = list(self._model.parameters())[0].device
            z_prev = th.zeros((env.num_envs, self._latent_dim), device=device)
        else:
            z_prev.index_fill_(0, th.where(obs['tick'].view(-1) == 0)[0], 0)
        new_z = (
            obs['tick'].long().remainder(self._encoder_interval).view(-1) == 0
        )
        new_z = new_z.unsqueeze(-1)

        def fwd_training(batch):
            if self._encoder_interval > 1:
                if batch['new_z'].any():
                    z_dist = self._encoder(batch, target=True)
                    z = z_dist.sample()
                    z = z_prev * th.logical_not(new_z) + z * new_z
                else:
                    z_dist = None
                    z = z_prev
            else:
                z_dist = self._encoder(batch, target=True)
                z = z_dist.rsample()
            if self._tanh_latents:
                dist = self._target_pi(
                    {'observation': batch['observation'], 'z': th.tanh(z)}
                )
            else:
                dist = self._target_pi(
                    {'observation': batch['observation'], 'z': z}
                )
            action = dist.rsample()
            return dist, action, z_dist, z

        if self.training:
            if len(self._fwd_batch) == 0:
                self._fwd_batch = {k: v for k, v in obs.items()}
                self._fwd_batch['new_z'] = new_z
                self._fwd_batch['z_prev'] = z_prev
            if self._init_fwd_graph:
                assert self._encoder_interval == 1
                log.info('Building forward graph')
                self._fwd_g, self._fwd_out = capture_graph(
                    lambda: fwd_training(self._fwd_batch)
                )
                self._init_fwd_graph = False
            if self._fwd_g:
                for k in obs.keys():
                    self._fwd_batch[k].copy_(obs[k])
                # self._fwd_batch['new_z'].copy_(new_z)
                self._fwd_batch['z_prev'].copy_(z_prev)
                self._fwd_g.replay()
                dist, action, z_dist, z = self._fwd_out
            else:
                for k in obs.keys():
                    self._fwd_batch[k] = obs[k]
                self._fwd_batch['new_z'] = new_z
                self._fwd_batch['z_prev'] = z_prev
                dist, action, z_dist, z = fwd_training(self._fwd_batch)
            extra = {
                'mu': dist.mean,
                'std': dist.stddev,
                'z_p': z_prev,
                'z_dist': z_dist,
                'z': z,
            }
        else:
            if new_z.any():
                z_dist = self._encoder(
                    {
                        'observation': obs['observation'],
                        'reference': obs['reference'],
                        'z_prev': z_prev,
                    }
                )
                z = z_dist.mean
                z = z_prev * th.logical_not(new_z) + z * new_z
            else:
                z_dist = None
                z = z_prev
            if self._tanh_latents:
                dist = self._model.pi(
                    {'observation': obs['observation'], 'z': th.tanh(z)}
                )
            else:
                dist = self._model.pi(
                    {'observation': obs['observation'], 'z': z}
                )
            action = dist.mean
            extra = {
                'mu': dist.mean,
                'std': dist.stddev,
                'z_p': z_prev,
                'z_dist': z_dist,
                'z': z,
            }

        env.ctx['z_prev'] = z
        return action, extra

    def step(
        self,
        env,
        obs,
        action: th.Tensor,
        extra: Any,
        result: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, List[Dict]],
    ) -> None:
        next_obs, reward, terminated, truncated, info = result
        done = terminated | truncated
        mean, stddev, z_prev = extra['mu'], extra['std'], extra['z_p']

        if env.ctx.get('id', None) is None:
            env.ctx['id'] = len(self._buffers)
            self._buffers.append(
                ReplayBuffer(
                    size=self._rp_size,
                    interleave=env.num_envs,
                    device=self._rpbuf_device,
                )
            )

        d = dict(
            action=action,
            reward=reward,
            pi_mean=mean,
            pi_stddev=stddev,
            terminal=done,
            timeout=truncated,
            z_prev=z_prev,
            not_done=th.logical_not(done),
        )
        for k in self._obs_keys:
            d[f'obs_{k}'] = obs[k]
            d[f'next_obs_{k}'] = next_obs[k]

        if self._multi_reward:
            assert isinstance(info, dict)
            assert 'rewards' in info
            keys = sorted(info['rewards'].keys())
            d['rewards'] = th.stack([info['rewards'][k] for k in keys], dim=-1)

        self._buffers[env.ctx['id']].put_row(d)

        self._n_steps += 1
        self._n_samples += done.nelement()
        buffer_size = sum((b.size for b in self._buffers))
        buffer_tlen = min((b.tlen for b in self._buffers))
        if not (
            buffer_size >= self._bsz * self._target_steps
            and buffer_tlen >= self._batch_traj_len
        ):
            return

        log.debug(
            f'Spent {time.perf_counter() - self._last_update_time}s collecting samples'
        )

        t = time.perf_counter()
        N, T = self._batch_num_traj, self._batch_traj_len
        if self._batch_num_traj != env.num_envs:
            assert len(self._buffers) == 1
            iters = (self._target_steps * self._bsz) // (
                env.num_envs * self._batch_traj_len
            )
            ups_per_iter = self._target_steps // iters
            for j in range(iters):
                batch = self._buffers[0].pop_rows_front(T)
                for i in range(ups_per_iter):
                    b = {
                        k: v.view((T, N * ups_per_iter) + v.shape[1:]).narrow(
                            1, i * N, N
                        )
                        for k, v in batch.items()
                    }
                    b = {
                        k: v.reshape((T * N,) + v.shape[2:])
                        for k, v in b.items()
                    }
                    if (
                        self._batch is None
                        or self._init_graph
                        or self._vup_g is None
                    ):
                        self._batch = b
                    else:
                        for k in b.keys():
                            self._batch[k].copy_(b[k])
                    self.update(self._batch)
        else:
            for i in range(self._target_steps):
                batch = self._buffers[i % len(self._buffers)].pop_rows_front(T)
                self.update(batch)

        log.debug(f'Updates done in {time.perf_counter()-t}s')
        for i, buf in enumerate(self._buffers):
            if buf.size > 0:
                log.info(f'!!! {buf.size} samples left in buffer {i}')
            buf.clear()

        # if self.n_updates % self._target_steps == 0:
        with th.no_grad():
            for tp, p in zip(
                self._target_encoder.parameters(),
                self._model.encoder.parameters(),
            ):
                tp.data.copy_(p.data)
            for tp, p in zip(
                self._target_pi.parameters(), self._model.pi.parameters()
            ):
                tp.data.copy_(p.data)

        self._last_update_time = time.perf_counter()

    def tbw_add_scalars(
        self,
        title: str,
        vals: th.Tensor,
        agg=['mean', 'min', 'max'],
        n_samples=None,
    ):
        if self.tbw is None:
            return
        data = {a: getattr(vals, a)() for a in agg}
        self.tbw.add_scalars(
            title,
            data,
            self._log_factor
            * (n_samples if n_samples is not None else self._n_samples),
        )

    def tbw_add_scalar(self, title: str, value: float, n_samples=None):
        if self.tbw is None:
            return
        self.tbw.add_scalar(
            title,
            value,
            self._log_factor
            * (n_samples if n_samples is not None else self._n_samples),
        )

    def _encoder(self, x, target=False):
        if target:
            z_dist = self._target_encoder(x)
        else:
            z_dist = self._model.encoder(x)
        if self._prior == 'ar1':
            alpha = self._ar1_alpha
            z_dist = D.Normal(z_dist.loc + alpha * x['z_prev'], z_dist.scale)
        return z_dist

    def _loss_bwd(self, batch):
        N, T = self._batch_num_traj, self._batch_traj_len
        obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
        obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
        reward = batch['reward']
        rewards = batch['rewards']
        pi_mean = batch['pi_mean']
        pi_stddev = batch['pi_stddev']
        action = batch['action']
        z_prev = batch['z_prev']
        timeout = batch['timeout']
        not_done = batch['not_done']

        try:
            self._optim._all_.zero_grad()
        except:
            self._optim.pi.zero_grad()
            self._optim.v.zero_grad()
            if self._popart:
                self._optim.popart.zero_grad()
        self._optim_lg.zero_grad()

        value = self._model.v(obs)
        with th.no_grad():
            next_value = self._model.v(obs_p)
        if self._popart:
            with th.no_grad():
                next_value, _ = self._model.popart(next_value)
            next_value = next_value.view(-1)

        # Bootstrap with current value function in timeout states and at the end
        # of the batch.
        if not self._multi_reward:
            ret = reward.clone()
            ret = ret + timeout * self._gamma * next_value
            ret = ret.view(T, N)
            ret[-1].add_(
                not_done.view(T, N)[-1]
                * self._gamma
                * next_value.view(T, N)[-1]
            )
            discounted_bwd_cumsum_(
                ret, discount=self._gamma, mask=not_done.view(T, N), dim=0
            )
        else:
            ret = rewards.clone()
            ret = ret + timeout.unsqueeze(-1) * self._gamma * next_value
            ret = ret.view(T, N, -1)
            ret[-1].add_(
                (
                    not_done.view(T, N, 1)[-1]
                    * self._gamma
                    * next_value.view(T, N, -1)[-1]
                )
            )
            discounted_bwd_cumsum_(
                ret,
                discount=self._gamma,
                mask=not_done.view(T, N, 1).repeat(1, 1, ret.shape[-1]),
                dim=0,
            )

        if self._popart:
            self._model.popart.update_parameters(ret.unsqueeze(-1))
            _, norm_value = self._model.popart(value)
            norm_value = norm_value.view(-1)
            ret = self._model.popart.normalize(ret)
            adv = ret.view(-1) - norm_value.detach()
            value = norm_value
        else:
            if not self._multi_reward:
                adv = ret.view(-1) - value.detach().view(-1)
            else:
                adv = ret.sum(dim=-1).view(-1) - value.detach().sum(dim=-1)

        v_loss = 0.5 * F.mse_loss(value.view(-1), ret.view(-1))

        eta = self._lg[0]
        alpha_mu = self._lg[1]
        alpha_sigma = self._lg[2]

        # Select 50% top advantages
        top_adv, top_adv_idx = th.topk(adv, (N * T) // 2, sorted=False)
        adv_mask = th.zeros_like(adv)
        # adv_mask[top_adv_idx] = 1
        adv_mask.index_fill_(0, top_adv_idx, 1)

        log_adv_sum = th.logsumexp(top_adv / eta, dim=0)
        eta_loss = eta * self._eps_eta + eta * (log_adv_sum - np.log(N * T))

        z_dist = self._encoder(
            {
                'observation': obs['observation'],
                'reference': obs['reference'],
                'z_prev': z_prev,
            }
        )
        if self._prior == 'normal':
            prior = D.Normal(0, 1)
            kl_reg = self._kl_reg
        elif self._prior == 'ar1':
            # AR(1) prior from Bohez et al., 2022
            alpha = self._ar1_alpha
            prior = D.Normal(
                alpha * z_prev, (1 - alpha) ** 2 * th.ones_like(z_prev)
            )
            kl_reg = self._kl_reg * (
                1
                - math.pow(
                    1 - min(1, 2 * self._n_samples / self._max_steps), 0.2
                )
            )
        kl_loss = (
            kl_reg
            * D.kl.kl_divergence(z_dist, prior)
            .view(T, N, -1)
            .sum(dim=-1)
            .sum(dim=0)
            .mean()
        )

        z = z_dist.rsample()
        if self._tanh_latents:
            z = th.tanh(z)

        if self._encoder_interval > 1:
            new_z = obs['tick'].remainder(self._encoder_interval) == 0
            new_z = new_z.view(T, N, 1)
            keep_z = th.logical_not(new_z)
            z = z.view(T, N, -1)
            z_p = th.zeros_like(z)
            # Constant z while new_z is False. Except for the first frame. We
            # could take those from the replay buffer as well.
            z_p[0].copy_(z[0])
            for t in range(1, T):
                z_p[t] = z[t] * new_z[t] + z[t - 1] * keep_z[t]
            z = z_p.view(T * N, -1)

        dist = self._model.pi({'observation': obs['observation'], 'z': z})
        dist_old = D.Normal(pi_mean, pi_stddev)
        with th.no_grad():
            phi = th.exp((adv / eta) - log_adv_sum)
        pi_loss = -(adv_mask * (phi * dist.log_prob(action).sum(-1))).sum()

        if self._distributed:
            # Gradients will be averaged in distributed traininig. This is
            # correct for all losses (which are already averaged over the
            # equal-sized mini-batches) except the policy loss (which is a sum).
            # We'll fix this by scaling the policy loss accordingly beforehand.
            ws = thd.get_world_size()
            logged_pi_loss = pi_loss
            pi_loss = pi_loss * ws
        else:
            logged_pi_loss = pi_loss

        mean_diff = dist.mean - dist_old.mean
        mu_kl = 0.5 * th.sum(
            mean_diff.square() * dist_old.stddev.reciprocal(), dim=-1
        )
        trace = (dist_old.stddev / dist.stddev).sum(dim=-1)
        det_std_log = th.sum(dist.stddev.log(), dim=-1)
        det_std_old_log = th.sum(dist_old.stddev.log(), dim=-1)
        sigma_kl = 0.5 * (
            trace - action.shape[-1] + det_std_log - det_std_old_log
        )
        alpha_mu_loss = (
            alpha_mu * (self._eps_alpha_mu - mu_kl.detach())
            + alpha_mu.detach() * mu_kl
        )
        alpha_mu_loss = alpha_mu_loss.mean()
        alpha_sigma_loss = (
            alpha_sigma * (self._eps_alpha_sigma - sigma_kl.detach())
            + alpha_sigma.detach() * sigma_kl
        )
        alpha_sigma_loss = alpha_sigma_loss.mean()

        loss = (
            pi_loss
            + v_loss
            + eta_loss
            + alpha_mu_loss
            + alpha_sigma_loss
            + kl_loss
        )
        loss.backward()

        return (
            v_loss,
            adv,
            ret,
            pi_loss,
            eta_loss,
            alpha_mu_loss,
            alpha_sigma_loss,
            kl_loss,
            logged_pi_loss,
            dist,
        )

    def _update(self, batch) -> None:
        if self._init_graph:
            log.info('Building update graph')
            self._vup_g, self._vup_out = capture_graph(
                lambda: self._loss_bwd(batch)
            )
            self._init_graph = False
        if self._vup_g is not None:
            self._vup_g.replay()
            (
                v_loss,
                adv,
                ret,
                pi_loss,
                eta_loss,
                alpha_mu_loss,
                alpha_sigma_loss,
                kl_loss,
                logged_pi_loss,
                dist,
            ) = self._vup_out
        else:
            (
                v_loss,
                adv,
                ret,
                pi_loss,
                eta_loss,
                alpha_mu_loss,
                alpha_sigma_loss,
                kl_loss,
                logged_pi_loss,
                dist,
            ) = self._loss_bwd(batch)

        if self._distributed:
            ws = thd.get_world_size()
            for p in self._model.parameters():
                if p.grad is not None:
                    thd.all_reduce(p.grad)
                    p.grad.div_(ws)
            thd.all_reduce(self._lg.grad)
            self._lg.grad.div_(ws)
        self._optim._all_.step()
        self._optim_lg.step()

        # Clamp lagrange multipliers
        self._lg.data.clamp_(min=1e-8)

        if self.n_updates % self._target_steps != 0:
            return

        # First frame of trajectory is at beginning (if not marked done) or
        # after terminal frames.
        N, T = self._batch_num_traj, self._batch_traj_len
        done = batch['terminal']
        begins = th.zeros_like(batch['not_done'].view(T, N))
        begins[0] = batch['not_done'].view(T, N)[0]
        begins[1:] |= done.view(T, N)[:-1]
        # Per-episode discounted return is accumulated until first frame
        # TODO This is wrong, ret includes bootstrapping!
        ep_returns = ret[th.where(begins)].view(-1)
        # Undiscounted return is accumulated reward at last frame
        # ep_returns_undisc = batch['reward_acc'][th.where(done)].view(-1)
        self.tbw_add_scalars('Perf/Reward', batch['reward'])
        self.tbw_add_scalars('Perf/Return', ep_returns)
        # self.tbw_add_scalars('Perf/ReturnUndisc', ep_returns_undisc)
        self.tbw_add_scalar('Loss/Policy', logged_pi_loss.item())
        self.tbw_add_scalar('Loss/Value', v_loss.item())
        self.tbw_add_scalar('Loss/Eta', eta_loss.item())
        self.tbw_add_scalar('Loss/Alpha_mu', alpha_mu_loss.item())
        self.tbw_add_scalar('Loss/Alpha_sigma', alpha_sigma_loss.item())
        self.tbw_add_scalar('Loss/KL', kl_loss.item())
        self.tbw_add_scalar(
            "Health/Entropy", -dist.log_prob(batch['action']).mean()
        )
        lg = self._lg.cpu()
        eta, alpha_mu, alpha_sigma = lg[0].item(), lg[1].item(), lg[2].item()
        self.tbw_add_scalar('Health/Eta', eta)
        self.tbw_add_scalar('Health/Alpha_mu', alpha_mu)
        self.tbw_add_scalar('Health/Alpha_sigma', alpha_sigma)
        # self.tbw_add_scalar('Health/KL_mu', mu_kl.mean().item())
        # self.tbw_add_scalar('Health/KL_sigma', sigma_kl.mean().item())
        self.tbw_add_scalars('Health/Advantage', adv)
        # self.tbw_add_scalars('Health/TopAdvantage', top_adv)
        # self.tbw_add_scalars('Health/Phi', phi)
        self.tbw_add_scalar('Health/KL_encoder', kl_loss.item() / self._kl_reg)
        self.tbw_add_scalar('Episodes_in_batch', batch["terminal"].sum().item())

        log.info(
            f'Up {self._n_updates:>4} avg rew {batch["reward"].mean():+.02f}, avg return {ep_returns.mean():+.02f}, pi loss {logged_pi_loss.item():+.02f}, v loss {v_loss.item():+.02f}, entropy {-dist.log_prob(batch["action"]).mean().item():+.02f}, kl {kl_loss.item()/self._kl_reg:+0.2f} eta {eta:.02f}, alpha {alpha_mu:.02f}/{alpha_sigma:.02f} done {done.sum().item()}'
        )
