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

from hucc import ReplayBuffer
from hucc.agents import Agent
from hucc.agents.utils import discounted_bwd_cumsum_
from hucc.spaces import box_space, th_flatten

log = logging.getLogger(__name__)


class VMPOAgent(Agent):
    '''
    V-MPO agent from Song et al., 2019
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
        if not hasattr(model, 'v'):
            raise ValueError('Model needs "v" module')

        self._model = model
        self._optim = optim
        self._batch_num_traj = int(cfg.batch_num_traj)
        self._batch_traj_len = int(cfg.batch_traj_len)
        self._bsz = self._batch_num_traj * self._batch_traj_len  # in samples
        self._flatten_obs = bool(cfg.flatten_obs)
        self._gamma = float(cfg.gamma)
        self._epochs = float(cfg.epochs)
        self._target_steps = int(cfg.target_steps)
        self._lg = th.tensor(
            [float(cfg.eta), float(cfg.alpha_mu), float(cfg.alpha_sigma)],
            requires_grad=True,
            device=list(model.parameters())[0].device,
        )
        self._eta = self._lg[0]
        self._alpha_mu = self._lg[1]
        self._alpha_sigma = self._lg[2]
        self._eps_eta = float(cfg.eps_eta)
        self._eps_alpha_mu = float(cfg.eps_alpha_mu)
        self._eps_alpha_sigma = float(cfg.eps_alpha_sigma)
        self._multi_reward = bool(cfg.multi_reward)
        self._reward_keys: List[str] = []
        self._aux_loss = cfg.aux_loss
        self._aux_factor = float(cfg.aux_factor)
        self._ppo_style = bool(cfg.ppo_style)
        self._sample_reuse = float(cfg.sample_reuse)
        assert self._sample_reuse >= 0.0
        assert self._sample_reuse < 1.0
        self._offpolicy_correction = bool(cfg.offpolicy_correction)

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

        self._check_anomaly = cfg.get('detect_anomaly', False)

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

        self._target = deepcopy(self._model)
        # We'll never need gradients for the target network
        for param in self._target.parameters():
            param.requires_grad_(False)

        self._obs_space = env.observation_space
        if isinstance(self._obs_space, gym.spaces.Dict):
            self._obs_keys = list(self._obs_space.spaces.keys())
        else:
            self._obs_keys = []

        self.set_checkpoint_attr(
            '_model', '_target', '_optim', '_lg', '_optim_lg'
        )

        self._last_update_time = time.perf_counter()

    @property
    def supports_async_step(self):
        return True

    @staticmethod
    def effective_observation_space(env: gym.Env, cfg: DictConfig):
        if isinstance(env.observation_space, gym.spaces.Box):
            return env.observation_space

        pi_spaces = copy(env.observation_space.spaces)
        v_spaces = copy(env.observation_space.spaces)
        if 'aux' in pi_spaces:
            del pi_spaces['aux']
        if 'reference' in pi_spaces and cfg.get('reference_size', None):
            pi_spaces['reference'] = box_space((cfg.reference_size,))
            v_spaces['reference'] = box_space((cfg.reference_size,))
        return {
            'pi': gym.spaces.Dict(pi_spaces),
            'v': gym.spaces.Dict(v_spaces),
        }

    def action(self, env, obs) -> Tuple[th.Tensor, th.Tensor]:
        extra = {}
        if hasattr(self._model, 'ref_encoder'):
            enc = self._model.ref_encoder
            obs = copy(obs)
            if hasattr(enc, 'bottleneck'):
                enc.eval()
                with th.no_grad():
                    inp = (obs['reference'] - enc.input_mean) / enc.input_std
                    zs, xsq, _, _ = enc.bottleneck(
                        enc.encoders[0](inp.permute(0, 2, 1))
                    )
                xsq = [x.permute(0, 2, 1) for x in xsq]
                obs['reference'] = xsq[0]
            else:
                with th.no_grad():
                    obs['reference'] = enc(obs['reference'])

            extra['reference'] = obs['reference']

        mobs = th_flatten(self._obs_space, obs) if self._flatten_obs else obs

        if self.training:
            with th.no_grad():
                dist = self._target.pi(mobs)
            if isinstance(dist, tuple):
                dist, _ = dist
            action = dist.sample()
        else:
            with th.no_grad():
                dist = self._model.pi(mobs)
            if isinstance(dist, tuple):
                dist, _ = dist
            action = dist.mean
        extra['mu'] = dist.mean
        extra['std'] = dist.stddev
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
        mean, stddev = extra['mu'], extra['std']

        if env.ctx.get('id', None) is None:
            env.ctx['id'] = len(self._buffers)
            self._buffers.append(
                ReplayBuffer(
                    size=self._rp_size,
                    interleave=env.num_envs,
                    device=self._rpbuf_device,
                )
            )

        '''
        if hasattr(self._model, 'ref_encoder'):
            enc = self._model.ref_encoder
            next_obs = copy(next_obs)
            if hasattr(enc, 'bottleneck'):
                enc.eval()
                with th.no_grad():
                    _, _, iv, _ = enc(next_obs['reference'])
                next_obs['reference'] = iv['xs_quantised'][0].permute(0,2,1)
                obs = copy(obs)
                obs['reference'] = extra['reference']
            else:
                next_obs = copy(next_obs)
                with th.no_grad():
                    next_obs['reference'] = enc(next_obs['reference'])
                obs['reference'] = extra['reference']
        '''

        d = dict(
            action=action,
            reward=reward,
            pi_mean=mean,
            pi_stddev=stddev,
            terminal=done,
            timeout=truncated,
        )
        if self._obs_keys:
            for k in self._obs_keys:
                d[f'obs_{k}'] = obs[k]
                d[f'next_obs_{k}'] = next_obs[k]
            if hasattr(self._model, 'ref_encoder'):
                d['obs_reference'] = extra['reference']
        else:
            d['obs'] = obs
            d['next_obs'] = next_obs

        if self._multi_reward:
            assert 'rewards' in info
            if not self._reward_keys:
                self._reward_keys = list(sorted(info['rewards'].keys()))
            d['rewards'] = th.stack(
                [info['rewards'][k] for k in self._reward_keys], dim=-1
            )

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
                    self.update(b)
                if j >= iters * (1 - self._sample_reuse):
                    for i in range(T):
                        self._buffers[0].put_rows(
                            {k: v.clone() for k, v in batch.items()}
                        )
        else:
            for i in range(self._target_steps):
                buf = self._buffers[i % len(self._buffers)]
                batch = buf.pop_rows_front(T)
                self.update(batch)

                if i >= self._target_steps * (1 - self._sample_reuse):
                    buf.put_rows({k: v.clone() for k, v in batch.items()})

        log.debug(f'Updates done in {time.perf_counter()-t}s')
        if self._sample_reuse == 0:
            for i, buf in enumerate(self._buffers):
                if buf.size > 0:
                    log.info(f'!!! {buf.size} samples left in buffer {i}')
                buf.clear()
        # if self.n_updates % self._target_steps == 0:
        with th.no_grad():
            for tp, p in zip(
                self._target.parameters(), self._model.parameters()
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

    def _update(self, batch) -> None:
        if self._ppo_style:
            self._update_ppo_style(batch)
        else:
            self._update_default(batch)

    def _update_default(self, batch) -> None:
        for p in self._model.parameters():
            mdevice = p.device
            break

        N, T = self._batch_num_traj, self._batch_traj_len
        for k in batch.keys():
            batch[k] = batch[k].to(mdevice)

        batch = dict(batch)
        reward = batch['reward']
        done = batch['terminal']
        timeout = batch['timeout']
        not_done = th.logical_not(done)

        if hasattr(self._model, 'ref_encoder'):
            # Ensure consistent reference encoding: take from succeeding
            # observation and re-compute for the last time-step
            reference_shifted = batch['obs_reference'].view(T, N, -1)[1:]
            last_ref = batch['next_obs_reference'].view(T, N, -1)[-1]
            enc = self._model.ref_encoder
            if hasattr(enc, 'bottleneck'):
                enc.eval()
                with th.no_grad():
                    inp = (last_ref - enc.input_mean) / enc.input_std
                    zs, xsq, _, _ = enc.bottleneck(
                        enc.encoders[0](inp.permute(0, 2, 1))
                    )
                xsq = [x.permute(0, 2, 1) for x in xsq]
                last_ref_encoded = xsq[0]
            else:
                with th.no_grad():
                    last_ref_encoded = enc(last_ref)

            batch['next_obs_reference'] = th.cat(
                [reference_shifted, last_ref_encoded.unsqueeze(0)], dim=0
            ).view(T * N, -1)

        if self._obs_keys:
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            if self._flatten_obs:
                obs = th_flatten(self._obs_space, obs)
                obs_p = th_flatten(self._obs_space, obs_p)
        else:
            obs = batch['obs']
            obs_p = batch['next_obs']

        try:
            self._optim._all_.zero_grad()
        except:
            self._optim.pi.zero_grad()
            self._optim.v.zero_grad()
            if self._popart:
                self._optim.popart.zero_grad()
        self._optim_lg.zero_grad()

        if self._obs_keys:
            v_in = {k: th.cat([obs[k], obs_p[k]], dim=0) for k in obs.keys()}
        else:
            v_in = th.cat([obs, obs_p], dim=0)
        v_out = self._model.v(v_in)
        if isinstance(v_out, tuple):
            v_out, aux_out_v = v_out
            aux_out_v = aux_out_v.chunk(2)[
                0
            ]  # only interested in current frame
        else:
            aux_out_v = None
        value, next_value = v_out.chunk(2)
        next_value = next_value.detach()
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
            ret = batch['rewards'].clone()
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

        # Select 50% top advantages
        top_adv, top_adv_idx = th.topk(adv, adv.numel() // 2, sorted=False)
        adv_mask = th.zeros_like(adv)
        adv_mask[top_adv_idx] = 1

        dist = self._model.pi(obs)
        if isinstance(dist, tuple):
            dist, aux_out_pi = dist
        else:
            aux_out_pi = None
        action = batch['action']
        dist_old = D.Normal(batch['pi_mean'], batch['pi_stddev'])

        if self._offpolicy_correction:
            with th.no_grad():
                old_log_prob = dist_old.log_prob(action)
                iw = (dist.log_prob(action) - old_log_prob).clamp(max=0)
            ifac = (iw + old_log_prob).sum(dim=-1)
            log_adv_sum = th.logsumexp(
                ifac[top_adv_idx] + top_adv / self._eta, dim=0
            )
        else:
            log_adv_sum = th.logsumexp(top_adv / self._eta, dim=0)

        with th.no_grad():
            if self._offpolicy_correction:
                phi = th.exp(ifac + (adv / self._eta) - log_adv_sum)
            else:
                phi = th.exp((adv / self._eta) - log_adv_sum)
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

        eta_loss = self._eta * self._eps_eta + self._eta * (
            log_adv_sum - np.log(top_adv.numel())
        )

        mean_diff = dist.mean - dist_old.mean
        mu_kl = 0.5 * th.sum(
            mean_diff.square() * dist_old.stddev.reciprocal(), dim=-1
        )
        trace = (dist_old.stddev / dist.stddev).sum(dim=-1)
        det_std_log = th.sum(dist.stddev.log(), dim=-1)
        det_std_old_log = th.sum(dist_old.stddev.log(), dim=-1)
        sigma_kl = 0.5 * (
            trace - dist.mean.shape[-1] + det_std_log - det_std_old_log
        )
        alpha_mu_loss = (
            self._alpha_mu * (self._eps_alpha_mu - mu_kl.detach())
            + self._alpha_mu.detach() * mu_kl
        )
        alpha_mu_loss = alpha_mu_loss.mean()
        alpha_sigma_loss = (
            self._alpha_sigma * (self._eps_alpha_sigma - sigma_kl.detach())
            + self._alpha_sigma.detach() * sigma_kl
        )
        alpha_sigma_loss = alpha_sigma_loss.mean()

        loss = pi_loss + v_loss + eta_loss + alpha_mu_loss + alpha_sigma_loss

        aux_loss_pi = None
        if self._aux_loss and aux_out_pi is not None:
            if self._aux_loss == 'l1':
                aux_loss_pi = (
                    (aux_out_pi.view(-1) - obs['aux'].view(-1)).abs().mean()
                )
            elif self._aux_loss == 'l2':
                aux_loss_pi = F.mse_loss(
                    aux_out_pi.view(-1), obs['aux'].view(-1)
                )
            else:
                raise ValueError('Unkown aux loss')
            loss = loss + self._aux_factor * aux_loss_pi
        aux_loss_v = None
        if self._aux_loss and aux_out_v is not None:
            if self._aux_loss == 'l1':
                aux_loss_v = (
                    (aux_out_v.view(-1) - obs['aux'].view(-1)).abs().mean()
                )
            elif self._aux_loss == 'l2':
                aux_loss_v = F.mse_loss(aux_out_v.view(-1), obs['aux'].view(-1))
            else:
                raise ValueError('Unkown aux loss')
            loss = loss + self._aux_factor * aux_loss_v

        loss.backward()
        if self._distributed:
            ws = thd.get_world_size()
            for p in self._model.parameters():
                if p.grad is not None:
                    thd.all_reduce(p.grad)
                    p.grad.div_(ws)
            thd.all_reduce(self._lg.grad)
            self._lg.grad.div_(ws)
        try:
            self._optim._all_.step()
        except:
            self._optim.pi.step()
            self._optim.v.step()
            if self._popart:
                self._optim.popart.step()
        self._optim_lg.step()

        # Clamp lagrange multipliers
        self._lg.data.clamp_(min=1e-8)

        if self.n_updates % self._target_steps != 0:
            return

        # First frame of trajectory is at beginning (if not marked done) or
        # after terminal frames.
        begins = th.zeros_like(not_done.view(T, N))
        begins[0] = not_done.view(T, N)[0]
        begins[1:] |= done.view(T, N)[:-1]
        # Per-episode discounted return is accumulated until first frame
        # TODO This is wrong, ret includes bootstrapping!
        ep_returns = ret[th.where(begins)].view(-1)
        # Undiscounted return is accumulated reward at last frame
        # ep_returns_undisc = batch['reward_acc'][th.where(done)].view(-1)
        self.tbw_add_scalars('Perf/Reward', reward)
        if self._multi_reward:
            for i, k in enumerate(self._reward_keys):
                self.tbw_add_scalars(f'Perf/Reward_{k}', batch['rewards'][:, i])
        self.tbw_add_scalars('Perf/Return', ep_returns)
        # self.tbw_add_scalars('Perf/ReturnUndisc', ep_returns_undisc)
        self.tbw_add_scalar('Loss/Policy', logged_pi_loss.item())
        self.tbw_add_scalar('Loss/Value', v_loss.item())
        self.tbw_add_scalar('Loss/Eta', eta_loss.item())
        self.tbw_add_scalar('Loss/Alpha_mu', alpha_mu_loss.item())
        self.tbw_add_scalar('Loss/Alpha_sigma', alpha_sigma_loss.item())
        if aux_loss_pi is not None:
            self.tbw_add_scalar('Loss/Aux_policy', aux_loss_pi.item())
        if aux_loss_v is not None:
            self.tbw_add_scalar('Loss/Aux_value', aux_loss_v.item())
        self.tbw_add_scalar("Health/Entropy", -dist.log_prob(action).mean())
        lg = self._lg.cpu()
        self.tbw_add_scalar('Health/Eta', lg[0].item())
        self.tbw_add_scalar('Health/Alpha_mu', lg[1].item())
        self.tbw_add_scalar('Health/Alpha_sigma', lg[2].item())
        self.tbw_add_scalar('Health/KL_mu', mu_kl.mean().item())
        self.tbw_add_scalar('Health/KL_sigma', sigma_kl.mean().item())
        self.tbw_add_scalars('Health/Advantage', adv)
        self.tbw_add_scalars('Health/TopAdvantage', top_adv)
        self.tbw_add_scalars('Health/Phi', phi)
        self.tbw_add_scalar('Episodes_in_batch', batch["terminal"].sum().item())

        aux_str = ''
        if aux_loss_pi is not None:
            aux_str += f' pi aux {aux_loss_pi.item():+.02f}'
        if aux_loss_v is not None:
            aux_str += f' v aux {aux_loss_v.item():+.02f}'
        log.info(
            f'Up {self._n_updates:>4} avg rew {reward.mean():+.02f}, avg return {ep_returns.mean():+.02f}, pi loss {logged_pi_loss.item():+.02f}, v loss {v_loss.item():+.02f}{aux_str}, entropy {-dist.log_prob(action).mean().item():+.02f}, eta {self._eta.item():.02f}, alpha {self._alpha_mu.item():.02f}/{self._alpha_sigma.item():.02f} done {done.sum().item()}'
        )

    def _update_ppo_style(self, batch) -> None:
        for p in self._model.parameters():
            mdevice = p.device
            break

        N, T = self._batch_num_traj, self._batch_traj_len
        for k in batch.keys():
            batch[k] = batch[k].to(mdevice)

        reward = batch['reward']
        done = batch['terminal']
        timeout = batch['timeout']
        not_done = th.logical_not(done)

        if self._obs_keys:
            obs = {k: batch[f'obs_{k}'] for k in self._obs_keys}
            obs_p = {k: batch[f'next_obs_{k}'] for k in self._obs_keys}
            if self._flatten_obs:
                obs = th_flatten(self._obs_space, obs)
                obs_p = th_flatten(self._obs_space, obs_p)
        else:
            obs = batch['obs']
            obs_p = batch['next_obs']

        if self._obs_keys:
            v_in = {k: th.cat([obs[k], obs_p[k]], dim=0) for k in obs.keys()}
        else:
            v_in = th.cat([obs, obs_p], dim=0)
        v_out = self._model.v(v_in)
        if isinstance(v_out, tuple):
            v_out, aux_out_v = v_out
            aux_out_v = aux_out_v.chunk(2)[
                0
            ]  # only interested in current frame
        else:
            aux_out_v = None
        value, next_value = v_out.chunk(2)
        next_value = next_value.detach()
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
            ret = batch['rewards'].clone()
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

        aux_loss_v = None
        if self._aux_loss and aux_out_v is not None:
            if self._aux_loss == 'l1':
                aux_loss_v = (
                    (aux_out_v.view(-1) - obs['aux'].view(-1)).abs().mean()
                )
            elif self._aux_loss == 'l2':
                aux_loss_v = F.mse_loss(aux_out_v.view(-1), obs['aux'].view(-1))
            else:
                raise ValueError('Unkown aux loss')
            v_loss = v_loss + self._aux_factor * aux_loss_v

        # Update value function once
        self._optim.v.zero_grad()
        if self._popart:
            self._optim.popart.zero_grad()
        v_loss.backward()
        if self._distributed:
            ws = thd.get_world_size()
            for p in self._model.v.parameters():
                if p.grad is not None:
                    thd.all_reduce(p.grad)
                    p.grad.div_(ws)
            if self._popart:
                for p in self._model.popart.parameters():
                    if p.grad is not None:
                        thd.all_reduce(p.grad)
                        p.grad.div_(ws)
        self._optim.v.step()
        if self._popart:
            self._optim.popart.step()

        # Update policy several times
        n_pi_updates = 4
        for _ in range(n_pi_updates):
            # Select random 50% of the 50% of advantages
            top_adv, top_adv_idx = th.topk(adv, adv.numel() // 2, sorted=False)
            rand_adv_idx = th.randperm(top_adv.numel(), device=top_adv.device)[
                : top_adv.numel() // 2
            ]
            top_adv = top_adv.index_select(0, rand_adv_idx)
            top_adv_idx = top_adv_idx.index_select(0, rand_adv_idx)
            adv_mask = th.zeros_like(adv)
            adv_mask[top_adv_idx] = 1

            log_adv_sum = th.logsumexp(top_adv / self._eta, dim=0)
            eta_loss = self._eta * self._eps_eta + self._eta * (
                log_adv_sum - np.log(top_adv.numel())
            )

            dist = self._model.pi(obs)
            if isinstance(dist, tuple):
                dist, aux_out_pi = dist
            else:
                aux_out_pi = None
            action = batch['action']
            dist_old = D.Normal(batch['pi_mean'], batch['pi_stddev'])
            with th.no_grad():
                phi = th.exp((adv / self._eta) - log_adv_sum)
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
                trace - dist.mean.shape[-1] + det_std_log - det_std_old_log
            )
            alpha_mu_loss = (
                self._alpha_mu * (self._eps_alpha_mu - mu_kl.detach())
                + self._alpha_mu.detach() * mu_kl
            )
            alpha_mu_loss = alpha_mu_loss.mean()
            alpha_sigma_loss = (
                self._alpha_sigma * (self._eps_alpha_sigma - sigma_kl.detach())
                + self._alpha_sigma.detach() * sigma_kl
            )
            alpha_sigma_loss = alpha_sigma_loss.mean()

            loss = pi_loss + eta_loss + alpha_mu_loss + alpha_sigma_loss

            self._optim.pi.zero_grad()
            self._optim_lg.zero_grad()
            loss.backward()
            if self._distributed:
                ws = thd.get_world_size()
                for p in self._model.parameters():
                    if p.grad is not None:
                        thd.all_reduce(p.grad)
                        p.grad.div_(ws)
                thd.all_reduce(self._lg.grad)
                self._lg.grad.div_(ws)
            self._optim.pi.step()
            self._optim_lg.step()

            # Clamp lagrange multipliers
            self._lg.data.clamp_(min=1e-8)

        if self.n_updates % self._target_steps != 0:
            return

        # First frame of trajectory is at beginning (if not marked done) or
        # after terminal frames.
        begins = th.zeros_like(not_done.view(T, N))
        begins[0] = not_done.view(T, N)[0]
        begins[1:] |= done.view(T, N)[:-1]
        # Per-episode discounted return is accumulated until first frame
        # TODO This is wrong, ret includes bootstrapping!
        ep_returns = ret[th.where(begins)].view(-1)
        # Undiscounted return is accumulated reward at last frame
        # ep_returns_undisc = batch['reward_acc'][th.where(done)].view(-1)
        self.tbw_add_scalars('Perf/Reward', reward)
        if self._multi_reward:
            for i, k in enumerate(self._reward_keys):
                self.tbw_add_scalars(f'Perf/Reward_{k}', batch['rewards'][:, i])
        self.tbw_add_scalars('Perf/Return', ep_returns)
        # self.tbw_add_scalars('Perf/ReturnUndisc', ep_returns_undisc)
        self.tbw_add_scalar('Loss/Policy', logged_pi_loss.item())
        self.tbw_add_scalar('Loss/Value', v_loss.item())
        self.tbw_add_scalar('Loss/Eta', eta_loss.item())
        self.tbw_add_scalar('Loss/Alpha_mu', alpha_mu_loss.item())
        self.tbw_add_scalar('Loss/Alpha_sigma', alpha_sigma_loss.item())
        if aux_loss_pi is not None:
            self.tbw_add_scalar('Loss/Aux_policy', aux_loss_pi.item())
        if aux_loss_v is not None:
            self.tbw_add_scalar('Loss/Aux_value', aux_loss_v.item())
        self.tbw_add_scalar("Health/Entropy", -dist.log_prob(action).mean())
        lg = self._lg.cpu()
        self.tbw_add_scalar('Health/Eta', lg[0].item())
        self.tbw_add_scalar('Health/Alpha_mu', lg[1].item())
        self.tbw_add_scalar('Health/Alpha_sigma', lg[2].item())
        self.tbw_add_scalar('Health/KL_mu', mu_kl.mean().item())
        self.tbw_add_scalar('Health/KL_sigma', sigma_kl.mean().item())
        self.tbw_add_scalars('Health/Advantage', adv)
        # self.tbw_add_scalars('Health/TopAdvantage', top_adv)
        self.tbw_add_scalars('Health/Phi', phi)
        self.tbw_add_scalar('Episodes_in_batch', batch["terminal"].sum().item())

        aux_str = ''
        if aux_loss_pi is not None:
            aux_str += f' pi aux {aux_loss_pi.item():+.02f}'
        if aux_loss_v is not None:
            aux_str += f' v aux {aux_loss_v.item():+.02f}'
        log.info(
            f'Up {self._n_updates:>4} avg rew {reward.mean():+.02f}, avg return {ep_returns.mean():+.02f}, pi loss {logged_pi_loss.item():+.02f}, v loss {v_loss.item():+.02f}{aux_str}, entropy {-dist.log_prob(action).mean().item():+.02f}, eta {self._eta.item():.02f}, alpha {self._alpha_mu.item():.02f}/{self._alpha_sigma.item():.02f} done {done.sum().item()}'
        )
