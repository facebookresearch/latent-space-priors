# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from copy import copy, deepcopy
from functools import partial
from typing import Any, Dict, List, Optional

import gym
import hydra
import numpy as np
import torch as th
import torch.distributions as D
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hucc.spaces import box_space, th_flatten

log = logging.getLogger(__name__)

_GRAPH_CAPTURING = False


class TransformedDistributionWithMean(D.TransformedDistribution):
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(
            base_distribution,
            transforms,
            validate_args if not _GRAPH_CAPTURING else False,
        )

    @property
    def mean(self):
        mu = self.base_dist.mean
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TransformDistribution(nn.Module):
    def __init__(self, transforms: List[D.Transform]):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        return TransformedDistributionWithMean(x, self.transforms)


class GaussianFromEmbedding(nn.Module):
    '''
    Computes a gaussian distribution from a vector input, using separate fully
    connnected layers for both mean and std (i.e. std is learned, too).
    '''

    def __init__(
        self,
        n_in: int,
        n_out: int,
        log_std_min: float = -2,
        log_std_max: float = 20,
        softplus_for_std: bool = False,
        smallinit: bool = False,
    ):
        super().__init__()
        self.mu_log_std = nn.Linear(n_in, n_out * 2)
        if smallinit:
            self.mu_log_std.weight.detach().div_(100)
            self.mu_log_std.bias.detach().div_(100)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.softplus_for_std = softplus_for_std

    def forward(self, input):
        mu, log_std = self.mu_log_std(input).chunk(2, -1)
        if not _GRAPH_CAPTURING and mu.isnan().any():
            import ipdb

            ipdb.set_trace()
        validate_args = None
        if _GRAPH_CAPTURING:
            validate_args = False
        if not self.softplus_for_std:
            log_std = log_std.clamp(self.log_std_min, self.log_std_max)
            return D.Normal(mu, log_std.exp(), validate_args=validate_args)
        else:
            return D.Normal(
                mu, F.softplus(log_std), validate_args=validate_args
            )


class GaussianFromMuLogStd(nn.Module):
    def __init__(self, log_std_min: float = -2, log_std_max: float = 20):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, input):
        mu, log_std = input.chunk(2, dim=-1)
        return D.Normal(
            mu, log_std.clamp(self.log_std_min, self.log_std_max).exp()
        )


class GaussianFromEmbeddingFixedStd(nn.Module):
    def __init__(self, n_in: int, n_out: int, std: float):
        super().__init__()
        self.std = std
        self.mu = nn.Linear(n_in, n_out)
        nn.init.zeros_(self.mu.bias)

    def forward(self, input):
        mu = self.mu(input)
        validate_args = None
        if _GRAPH_CAPTURING:
            validate_args = False
        return D.Normal(mu, self.std, validate_args=validate_args)


class GaussianSampleFromMuLogStdSoftplus(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp

    def forward(self, input):
        mu, log_std = input.chunk(2, dim=-1)
        if not self.training or self.temp == 0:
            return mu
        dist = D.Normal(mu, F.softplus(log_std) * self.temp)
        return dist.rsample()


class CategoricalFromEmbedding(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.logits = nn.Linear(n_in, n_out)

    def forward(self, input):
        logits = self.logits(input)
        return D.Categorical(logits=logits)


# From D2RL: Deep Dense Architectures in Reinforcement Learning
class SkipNetwork(nn.Module):
    def __init__(self, n_in: int, n_layers: int, n_hid: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, n_hid))
        n_hin = n_in + n_hid
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hin, n_hid))

    def forward(self, x):
        y = self.layers[0](x)
        y = F.relu(y, inplace=True)
        for l in self.layers[1:]:
            y = l(th.cat([y, x], dim=1))
            y = F.relu(y, inplace=True)
        return y


# SkipNetwork for compatibility with PhysicsVAE, where the network is trained on
# concatenated inputs.
class SkipNetworkD(SkipNetwork):
    def forward(self, x):
        x = th.cat(list(x.values()), dim=-1)
        return super().forward(x)


class GroupedLinear(nn.Module):
    def __init__(self, inp, outp, groups, bias=True):
        super().__init__()
        self.weight = nn.Parameter(th.Tensor(outp, inp))
        if bias:
            self.bias = nn.Parameter(th.Tensor(outp))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.register_buffer('wmask', th.zeros_like(self.weight))
        assert inp % groups == 0
        assert outp % groups == 0
        c_in = inp // groups
        c_out = outp // groups
        for g in range(groups):
            self.wmask[
                c_out * g : c_out * (g + 1), c_in * g : c_in * (g + 1)
            ].fill_(1)

        self.inp = inp
        self.outp = outp
        self.groups = groups

    def reset_parameters(self) -> None:
        from torch.nn import init

        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight * self.wmask, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(
            self.inp, self.outp, self.groups, self.bias is not None
        )


class SkipDoubleQNetwork(nn.Module):
    '''
    Model two Q-function networks inside a single one.
    '''

    def __init__(self, n_in: int, n_layers: int, n_hid: int, n_out: int = 1):
        super().__init__()
        layers = nn.ModuleList()
        self.l0 = nn.Linear(n_in, n_hid * 2)
        n_hin = n_in + n_hid
        for _ in range(n_layers - 1):
            layers.append(GroupedLinear(n_hin * 2, n_hid * 2, 2))
        self.layers = layers
        self.lN = GroupedLinear(n_hid * 2, n_out * 2, 2)
        th.fill_(self.lN.weight.detach(), 0)
        th.fill_(self.lN.bias.detach(), 0)

    def forward(self, x):
        y = self.l0(x)
        y = F.relu(y, inplace=True)
        for i, l in enumerate(self.layers):
            ys = y.chunk(2, dim=1)
            y = l(th.cat([ys[0], x, ys[1], x], dim=1))
            y = F.relu(y, inplace=True)
        y = self.lN(y)
        return y


class ComicEncoder(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.norm = nn.Sequential(nn.LayerNorm(n_in), nn.Tanh())
        # TODO use layer-norm after each linear?
        self.mlp = nn.Sequential(
            nn.Linear(n_in, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU()
        )

    def forward(self, x):
        x = self.norm(x)
        y = self.mlp(x)
        return th.cat([x, y], dim=-1)


class FlattenSpace(nn.Module):
    def __init__(self, space: gym.Space):
        super().__init__()
        self.space = space

    def forward(self, d):
        return th_flatten(self.space, d)

    def extra_repr(self) -> str:
        return str(self.space)


class DictSelect(nn.Module):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(self, d):
        return d[self.key]

    def extra_repr(self) -> str:
        return self.key


class MapToDictElement(nn.Module):
    def __init__(self, key: str, module: nn.Module):
        super().__init__()
        self.key = key
        self.m = module

    def forward(self, x):
        x = copy(x)
        x[self.key] = self.m(x[self.key])
        return x

    def extra_repr(self) -> str:
        return f'key={self.key}'


class SqueezeLastDim(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


class VMPOPopart(nn.Module):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()
        n_in = gym.spaces.flatdim(obs_space)
        n_out = gym.spaces.flatdim(action_space)
        self._ln = nn.Sequential(nn.LayerNorm(n_in), nn.Tanh())
        self._pi = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            GaussianFromEmbedding(
                n_in=256, n_out=n_out, softplus_for_std=True, smallinit=True
            ),
        )
        self._v = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.popart = PopArtLayer(256, 1, 1e-4)

    def pi(self, x):
        return self._pi(self._ln(x))

    def v(self, x):
        return self._v(self._ln(x))


# From https://github.com/aluscher/torchbeastpopart/blob/master/torchbeast/core/popart.py
class PopArtLayer(nn.Module):
    def __init__(self, input_features, output_features, beta=4e-4):
        self.beta = beta

        super().__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = th.nn.Parameter(
            th.Tensor(output_features, input_features)
        )
        self.bias = th.nn.Parameter(th.Tensor(output_features))

        self.register_buffer(
            'mu', th.zeros(output_features, requires_grad=False)
        )
        self.register_buffer(
            'sigma', th.ones(output_features, requires_grad=False)
        )

        self.reset_parameters()

    def reset_parameters(self):
        th.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = th.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            th.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        normalized_output = inputs.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)

        with th.no_grad():
            output = normalized_output * self.sigma + self.mu

        return output, normalized_output

    @th.no_grad()
    def normalize(self, inputs, task=None):
        if task is None:
            mu = self.mu[0]
            sigma = self.sigma[0]
        else:
            mu = (
                self.mu.expand(*inputs.shape, self.output_features)
                .gather(-1, task.unsqueeze(-1))
                .squeeze(-1)
            )
            sigma = (
                self.sigma.expand(*inputs.shape, self.output_features)
                .gather(-1, task.unsqueeze(-1))
                .squeeze(-1)
            )
        output = (inputs - mu) / sigma
        return output

    @th.no_grad()
    def update_parameters(self, vs, task=None):
        # task is opional one-hot encoding if num_outputs > 1
        oldmu = self.mu
        oldsigma = self.sigma

        if task is None:
            n = vs.numel()
        else:
            vs = vs * task
            n = task.sum((0, 1))
        mu = vs.sum((0, 1)) / n
        nu = th.sum(vs**2, (0, 1)) / n
        sigma = th.sqrt(nu - mu**2)
        sigma = th.clamp(sigma, min=1e-2, max=1e6)

        mu[th.isnan(mu)] = self.mu[th.isnan(mu)]
        sigma[th.isnan(sigma)] = self.sigma[th.isnan(sigma)]

        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma

        self.weight.data = (self.weight.t() * oldsigma / self.sigma).t()
        self.bias.data = (oldsigma * self.bias + oldmu - self.mu) / self.sigma


class AuxRegressionHead(nn.Module):
    def __init__(self, n_in: int, n_out: int, main_head: nn.Module):
        super().__init__()
        self.main_head = main_head
        self.aux_head = nn.Sequential(
            nn.Linear(n_in, 512), nn.ReLU(), nn.Linear(512, n_out)
        )

    def forward(self, x):
        out1 = self.main_head(x)
        out2 = self.aux_head(x)
        return out1, out2


class Ensure2D(nn.Module):
    def forward(self, x):
        if x.ndim == 1:
            return x.view(-1, 1)
        return x


def shorthand(name, *args, **kwargs):
    def inner(fn):
        if name in FactoryType._shorthands:
            raise ValueError(f'Duplicate shorthand: {name}')
        FactoryType._shorthands[name] = partial(fn, *args, **kwargs)
        return fn

    return inner


class FactoryType(type):
    _shorthands = {}

    def __getattr__(self, name):
        if name in self._shorthands:
            return self._shorthands[name]
        raise AttributeError(
            f'type object "{self.__name__}" has no attribute "{name}"'
        )


class Factory(metaclass=FactoryType):
    @staticmethod
    def make(name: str, *args, **kwargs):
        return getattr(Factory, name)(*args, **kwargs)

    @staticmethod
    def identity(*args, **kwargs):
        return nn.Identity()

    @staticmethod
    @shorthand('qd_d2rl_256', n_hid=256, flatten_input=False)
    @shorthand('qd_d2rl_512', n_hid=512, flatten_input=False)
    @shorthand('qd_d2rl_1024', n_hid=1024, flatten_input=False)
    @shorthand('qd_d2rl_d_256', n_hid=256, flatten_input=True)
    @shorthand('qd_d2rl_d_512', n_hid=512, flatten_input=True)
    @shorthand('qd_d2rl_d_1024', n_hid=1024, flatten_input=True)
    def qd_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
        n_emb: int = 8,  # embedding size for discrete spaces
    ) -> nn.Module:
        ms: List[nn.Module] = []
        if not flatten_input:
            n_in = gym.spaces.flatdim(obs_space) + gym.spaces.flatdim(
                action_space
            )
            ms.append(SkipDoubleQNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
            return nn.Sequential(*ms)

        n_in = gym.spaces.flatdim(action_space)
        obs_spaces: Dict[str, gym.spaces.Space] = {}
        for k, v in obs_space.spaces.items():
            if isinstance(v, gym.spaces.Discrete):
                ms.append(MapToDictElement(k, nn.Embedding(v.n, n_emb)))
                obs_spaces[k] = box_space((n_emb,))
            else:
                obs_spaces[k] = v

        n_in = gym.spaces.flatdim(
            gym.spaces.Dict(obs_spaces)
        ) + gym.spaces.flatdim(action_space)
        if isinstance(action_space, gym.spaces.Dict):
            joint_space = gym.spaces.Dict(
                dict(**action_space.spaces, **obs_spaces)
            )
        else:
            joint_space = gym.spaces.Dict(
                dict(action=action_space, **obs_spaces)
            )
        ms.append(FlattenSpace(joint_space))
        ms.append(SkipDoubleQNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('pi_d2rl_256', n_hid=256, flatten_input=False)
    @shorthand('pi_d2rl_512', n_hid=512, flatten_input=False)
    @shorthand('pi_d2rl_1024', n_hid=1024, flatten_input=False)
    @shorthand('pi_d2rl_d_256', n_hid=256, flatten_input=True)
    @shorthand('pi_d2rl_d_512', n_hid=512, flatten_input=True)
    @shorthand('pi_d2rl_d_1024', n_hid=1024, flatten_input=True)
    @shorthand(
        'pi_d2rl_d_256_notanh',
        n_hid=256,
        flatten_input=True,
        tanh_transform=False,
    )
    def pi_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
        tanh_transform: bool = True,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_in = gym.spaces.flatdim(obs_space)
        n_out = gym.spaces.flatdim(action_space)
        if flatten_input:
            ms.append(FlattenSpace(obs_space))
        ms.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        ms.append(
            GaussianFromEmbedding(
                n_in=n_hid, n_out=n_out, log_std_min=-5, log_std_max=2
            )
        )
        if tanh_transform:
            ms.append(TransformDistribution([D.TanhTransform(cache_size=1)]))
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('pi_comic_d2rl_gs_d_1024', n_hid=1024)
    @shorthand('pi_comic_d2rl_gs_ln_d_1024', n_hid=1024, layer_norm=True)
    @shorthand('pi_comic_d2rl_gs_ln_d_256', n_hid=256, layer_norm=True)
    @shorthand(
        'pi_comic_d2rl_gs_vc_ln_d_1024',
        n_hid=1024,
        layer_norm=True,
        variable_clip=True,
    )
    @shorthand(
        'pi_comic_d2rl_gs_vc_ln_d_256',
        n_hid=256,
        layer_norm=True,
        variable_clip=True,
    )
    def pi_comic_d2rl_gs(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        n_clipembed: int = 30,
        layer_norm: bool = False,
        variable_clip: bool = False,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_out = gym.spaces.flatdim(action_space)
        obs_spaces = deepcopy(obs_space.spaces)
        if variable_clip:
            for varf in ('clip_id', 'time'):
                if varf in obs_spaces:
                    del obs_spaces[varf]
        for k in ('clip_id', 'label'):
            if k in obs_spaces:
                if k == 'clip_id':
                    emb: nn.Module = nn.Embedding(obs_spaces[k].n, n_clipembed)
                else:
                    emb = nn.Sequential(
                        Ensure2D(),
                        nn.EmbeddingBag(
                            obs_spaces[k].n, n_clipembed, padding_idx=0
                        ),
                    )
                ms.append(MapToDictElement(k, emb))
                obs_spaces[k] = box_space((n_clipembed,))

        mobs_space = gym.spaces.Dict(obs_spaces)
        n_in = gym.spaces.flatdim(mobs_space)
        ms.append(FlattenSpace(mobs_space))
        if layer_norm:
            ms.append(nn.LayerNorm(n_in))
            ms.append(nn.Tanh())
        ms.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        ms.append(
            GaussianFromEmbedding(
                n_in=n_hid, n_out=n_out, softplus_for_std=True, smallinit=True
            )
        )
        return nn.Sequential(*ms)

    def vmpo_popart(obs_space: gym.Space, action_space: gym.Space) -> nn.Module:
        return VMPOPopart(obs_space, action_space)

    @staticmethod
    @shorthand('popart_1024', n_in=1024)
    def popart(
        osb_space: gym.Space,
        action_space: gym.Space,
        n_in: int = 256,
        n_out: int = 1,
        beta: float = 1e-4,
    ):
        return PopArtLayer(n_in, n_out, beta)

    @staticmethod
    @shorthand('v_d2rl_256', n_hid=256, flatten_input=False)
    @shorthand('v_d2rl_512', n_hid=512, flatten_input=False)
    @shorthand('v_d2rl_1024', n_hid=1024, flatten_input=False)
    @shorthand('v_d2rl_d_256', n_hid=256, flatten_input=True)
    @shorthand('v_d2rl_d_512', n_hid=512, flatten_input=True)
    @shorthand('v_d2rl_d_1024', n_hid=1024, flatten_input=True)
    @shorthand(
        'v_d2rl_ln_d_1024', n_hid=1024, flatten_input=True, layer_norm=True
    )
    @shorthand(
        'v_d2rl_ln_256_latent',
        n_hid=256,
        flatten_input=False,
        layer_norm=True,
        output_latent=True,
    )
    @shorthand(
        'v_d2rl_ln_1024_latent',
        n_hid=1024,
        flatten_input=False,
        layer_norm=True,
        output_latent=True,
    )
    def v_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        flatten_input: bool,
        layer_norm: bool = False,
        output_latent: bool = False,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        n_in = gym.spaces.flatdim(obs_space)
        if flatten_input:
            ms.append(FlattenSpace(obs_space))
        if layer_norm:
            ms.append(nn.LayerNorm(n_in))
            ms.append(nn.Tanh())
        ms.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        if not output_latent:
            out = nn.Linear(n_hid, 1)
            th.fill_(out.weight.detach(), 0)
            th.fill_(out.bias.detach(), 0)
            ms.append(out)
            ms.append(SqueezeLastDim())
        return nn.Sequential(*ms)

    @staticmethod
    @shorthand('v_comic_d2rl_d_256', n_hid=256)
    @shorthand('v_comic_d2rl_d_512', n_hid=512)
    @shorthand('v_comic_d2rl_d_1024', n_hid=1024)
    @shorthand('v_comic_d2rl_ln_d_256', n_hid=256, layer_norm=True)
    @shorthand('v_comic_d2rl_ln_d_512', n_hid=512, layer_norm=True)
    @shorthand('v_comic_d2rl_ln_d_1024', n_hid=1024, layer_norm=True)
    @shorthand(
        'v_comic_d2rl_ln_d_1024_latent',
        n_hid=1024,
        layer_norm=True,
        output_latent=True,
    )
    def v_comic_d2rl(
        obs_space: gym.Space,
        action_space: gym.Space,
        n_hid: int,
        n_clipembed: int = 30,
        n_refembed: int = 128,
        layer_norm: bool = False,
        aux_head: bool = False,
        n_aux: int = 3,
        output_latent: bool = False,
        reward_terms: int = 1,
        zero_out: bool = True,
    ) -> nn.Module:
        ms: List[nn.Module] = []
        obs_spaces = deepcopy(obs_space.spaces)
        if isinstance(obs_space, gym.spaces.Dict):
            for k in ('clip_id', 'label'):
                if k in obs_spaces:
                    if k == 'clip_id':
                        emb: nn.Module = nn.Embedding(
                            obs_spaces[k].n, n_clipembed
                        )
                    else:
                        emb = nn.Sequential(
                            Ensure2D(),
                            nn.EmbeddingBag(
                                obs_spaces[k].n, n_clipembed, padding_idx=0
                            ),
                        )
                    ms.append(MapToDictElement(k, emb))
                    obs_spaces[k] = box_space((n_clipembed,))

        n_refwin = 1  # need to manually adjust aux outputs

        mobs_space = gym.spaces.Dict(obs_spaces)
        n_in = gym.spaces.flatdim(mobs_space)
        if isinstance(obs_space, gym.spaces.Dict):
            ms.append(FlattenSpace(mobs_space))
        if layer_norm:
            ms.append(nn.LayerNorm(n_in))
            ms.append(nn.Tanh())

        if aux_head:
            value_head = []
        else:
            value_head = ms  # XXX hack hack

        value_head.append(SkipNetwork(n_in=n_in, n_layers=4, n_hid=n_hid))
        if not output_latent:
            out = nn.Linear(n_hid, reward_terms)
            if zero_out:
                th.fill_(out.weight.detach(), 0)
            th.fill_(out.bias.detach(), 0)
            value_head.append(out)
            value_head.append(SqueezeLastDim())

        if aux_head:
            ms.append(
                AuxRegressionHead(
                    n_in=n_in,
                    n_out=n_aux * n_refwin,
                    main_head=nn.Sequential(*value_head),
                )
            )

        return nn.Sequential(*ms)

    @shorthand('convvae_4x64_32', block_width=64, block_depth=4, n_latent=32)
    def convvae(
        obs_space: gym.Space,
        action_space: gym.Space,
        seq_len: int,
        block_width: int,
        block_depth: int,
        n_latent: int,
        tanh_latents: bool = False,
    ) -> nn.Module:
        from hucc.models.convvae import ConvVAE

        n_in = gym.spaces.flatdim(obs_space)
        return ConvVAE(
            input_shape=(seq_len, n_in),
            block_width=block_width,
            block_depth=block_depth,
            n_latent=n_latent,
            tanh_latents=tanh_latents,
        )

    def simple_prior(
        obs_space: gym.Space,
        action_space: gym.Space,
        vae: DictConfig,
        level: int,
        labels: Optional[str] = None,  # (None, 'sequence', 'frame')
        n_ctx: int = 64,  # TODO dataset window size? // downs_t[0]
        width: int = 512,
        depth: int = 8,
        conditioner: str = 'default',
        conditioner_past_ctx: int = 1,
        **kwargs,
    ) -> nn.Module:
        from hucc.models.prior import SimplePrior

        if kwargs.get('vae_input_dim', None):
            vae_obs_space = box_space((kwargs['vae_input_dim'],))
            vae_m = make_model(vae, vae_obs_space, action_space)
        else:
            vae_m = make_model(vae, obs_space, action_space)

        # small_prior, small_upsampler
        init_scale = 0.7
        prior_kwargs = dict(
            input_shape=(n_ctx,),
            bins=vae_m.latent_dim,
            width=width,
            depth=depth,
            heads=kwargs.get('heads', 1),
            attn_order=0,
            blocks=1,
            spread=None,
            attn_dropout=kwargs.get('attn_dropout', 0.0),
            resid_dropout=kwargs.get('resid_dropout', 0.0),
            emb_dropout=kwargs.get('emb_dropout', 0.0),
            zero_out=False,
            res_scale=False,
            pos_init=False,
            init_scale=init_scale,
            m_attn=0.25,
            m_mlp=1.0,
            # disable in eval...?
            checkpoint_res=1,
            checkpoint_attn=0,
            checkpoint_mlp=0,
            entropy_reg=kwargs.get('entropy_reg', 0),
            tanh_output=kwargs.get('tanh_output', False),
            mixture_size=kwargs.get('mixture_size', 1),
        )
        x_cond_kwargs = dict(
            out_width=width,
            init_scale=init_scale,
            width=512,
            depth=4,
            m_conv=1.0,
            dilation_growth_rate=1,
            dilation_cycle=None,
            zero_out=False,
            res_scale=False,
            checkpoint_res=1,
        )
        if conditioner == 'simple':
            x_cond_kwargs['past_ctx'] = conditioner_past_ctx
        y_cond_kwargs = dict(  # for conditioning on labels
            out_width=width,
            init_scale=init_scale,
            y_bins=action_space.n if labels else 0,
            resolution=labels,
        )
        prime_kwargs = dict(
            use_tokens=False, prime_loss_fraction=0.1, n_tokens=0, bins=0
        )

        prior = SimplePrior(
            z_shapes=[(n_ctx,)],
            l_bins=vae_m.latent_dim,
            continuous=True,
            encoder=vae_m.encode,
            decoder=vae_m.decode,
            level=level,
            downs_t=[0],
            strides_t=[1],
            labels=labels,
            conditioner=conditioner,
            prior_kwargs=prior_kwargs,
            x_cond_kwargs=x_cond_kwargs,
            y_cond_kwargs=y_cond_kwargs,
            prime_kwargs=prime_kwargs,
            copy_input=False,
            labels_v3=False,
            merged_decoder=True,
            single_enc_dec=False,
            deterministic_z=kwargs.get('deterministic_z', True),
            deterministic_tgt=kwargs.get('deterministic_tgt', True),
            deterministic_cond=kwargs.get('deterministic_cond', True),
            bypass_encoder=kwargs.get('bypass_encoder', False),
        )

        # XXX someone has to hold the reference...
        prior.vae = vae_m
        return prior

    def gaussian_sample_softplus(
        obs_space: gym.Space,
        action_space: gym.Space,
        temp: float = 1.0,
        with_tanh: bool = False,
    ):
        if with_tanh:
            return nn.Sequential(
                GaussianSampleFromMuLogStdSoftplus(temp), nn.Tanh()
            )
        else:
            return GaussianSampleFromMuLogStdSoftplus(temp)


def make_model(
    cfg: DictConfig, obs_space: gym.Space, action_space: gym.Space
) -> nn.Module:
    if isinstance(cfg, str):
        return Factory.make(cfg, obs_space, action_space)
    elif isinstance(cfg, DictConfig):
        if '_obs_space_' in cfg:
            obs_space = hydra.utils.instantiate(cfg['_obs_space_'])
        if 'name' in cfg or '_target_' in cfg:
            name = cfg.get('_target_', cfg.get('name'))
            args = {
                k: v
                for k, v in cfg.items()
                if k != 'name' and not k.startswith('_')
            }
            model = Factory.make(name, obs_space, action_space, **args)
        else:
            models: Dict[str, nn.Module] = {}
            for k, v in cfg.items():
                if v is not None and not k.startswith('_'):
                    models[k] = make_model(v, obs_space, action_space)
            model = nn.ModuleDict(models)

        if '_init_from_' in cfg:
            log.info(f'Loading model from {cfg["_init_from_"]}')
            with open(cfg['_init_from_'], 'rb') as fd:
                data = th.load(fd, map_location='cpu')
            if 'model' in data:
                data = data['model']
            elif '_model' in data:
                data = data['_model']

            if '_init_map_' in cfg:
                for k, v in cfg._init_map_.items():
                    data = {dk.replace(v, k): dv for dk, dv in data.items()}

            missing, unexpected = model.load_state_dict(data, strict=False)
            if missing:
                if missing == ['input_mean', 'input_std']:
                    log.warning(
                        'input mean and std missing from data, I hope you\'ll apply them later?'
                    )
                else:
                    raise ValueError(f'Missing keys in data: {missing}')

            # XXX Hack for older VQVAE checkpoints
            if 'data_mean' in data:
                assert 'data_std' in data
                log.info('Copy input stats from checkpoint')
                model.input_mean.copy_(data['data_mean'])
                model.input_std.copy_(data['data_std'])
        return model
    else:
        raise ValueError(
            f'Can\'t handle model config: {cfg.pretty(resolve=True)}'
        )
