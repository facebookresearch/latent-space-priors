# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

import torch as th


def discounted_bwd_cumsum_(
    x: th.Tensor,
    discount: float,
    mask: Optional[th.Tensor] = None,
    dim: int = -1,
) -> th.Tensor:
    '''
    In-place computation of discounted cumulative sum, applied backwards (from n
    to 0) along a given dimension. If mask is provided, do not propagate values
    where it is 0; it is expected to be binary.
    '''
    if mask is not None and x.shape != mask.shape:
        raise ValueError(
            f'Input and mask are required to have the same shape ({x.shape} != {mask.shape})'
        )
    n = x.shape[dim]
    if mask is not None:
        for i in range(n - 2, -1, -1):
            x.select(dim, i).add_(
                mask.select(dim, i) * x.select(dim, i + 1), alpha=discount
            )
    else:
        for i in range(n - 2, -1, -1):
            x.select(dim, i).add_(x.select(dim, i + 1), alpha=discount)

    return x


def normalize(x: th.Tensor):
    '''
    Normalize x to zero mean and unit standard derivation.
    '''
    std, mu = th.std_mean(x)
    return (x - mu) / (std + 1e-6)


def normalize_masked(x: th.Tensor, mask: th.Tensor):
    '''
    Normalize x to zero mean and unit standard derivation.
    From rl-games
    '''
    sum_mask = mask.sum()
    x_mask = x * mask
    x_mean = x_mask.sum() / sum_mask
    min_sqr = (((x_mask) ** 2) / sum_mask).sum() - (
        (x_mask / sum_mask).sum()
    ) ** 2
    x_var = min_sqr * sum_mask / (sum_mask - 1)
    x_std = th.sqrt(x_var)
    return (x - x_mean) / (x_std + 1e-6)


_normalize_fn = normalize


def gae_advantage(
    reward: th.Tensor,
    value: th.Tensor,
    next_value: th.Tensor,
    gamma: float,
    lambd: float = 1,
    normalize: bool = True,
    mask: Optional[th.Tensor] = None,
    time_dim: int = 1,
):
    '''
    Generalized advantage estimation.  Expected shape of reward, value, mask is
    BxT -- unless time_dim is set to 0.
    '''
    assert reward.dim() == 2, f'Expected reward of BxT/TxB, got {reward.shape}'
    assert (
        value.shape == reward.shape
    ), f'Expected value shape of {reward.shape}, got {value.shape}'
    assert (
        next_value.shape == reward.shape
    ), f'Expected value shape of {reward.shape}, got {next_value.shape}'
    assert (
        mask is None or mask.shape == reward.shape
    ), f'Expected mask shape of {reward.shape}, got {mask.shape}'
    if mask is not None:
        deltas = reward + mask * gamma * next_value - value
        adv = discounted_bwd_cumsum_(
            deltas, gamma * lambd, mask=mask, dim=time_dim
        )
    else:
        deltas = reward + gamma * next_value - value
        adv = discounted_bwd_cumsum_(deltas, gamma * lambd, dim=time_dim)
    return _normalize_fn(adv) if normalize else adv


def batch_2to1(x: th.Tensor) -> th.Tensor:
    '''
    Flattens first two dimensions into a single one.
    '''
    s = list(x.shape)
    return x.reshape([s[0] * s[1]] + s[2:])


def clampt(x: th.Tensor, min: th.Tensor, max: th.Tensor) -> th.Tensor:
    return th.max(th.min(x, max), min)
