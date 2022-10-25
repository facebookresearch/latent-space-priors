# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch as th
from gym.spaces import Box, Dict, Discrete


def th_flatten(space, x) -> th.Tensor:
    '''
    Adapted from gym.spaces.flatten(); accounts for batch dimension.
    '''
    if isinstance(space, Box):
        return x.view(x.shape[0], -1)
    elif isinstance(space, Discrete):
        return x.view(x.shape[0], -1)
    elif isinstance(space, Dict):
        return th.cat(
            [th_flatten(s, x[key]) for key, s in space.spaces.items()], 1
        )
    else:
        raise NotImplementedError(
            f'No flatten implementation for {type(space)}'
        )


def th_unflatten(space, x: th.Tensor):
    '''
    Adapted from gym.spaces.unflatten().
    '''
    if isinstance(space, Box):
        return x.view(x.shape[0], *space.shape)
    elif isinstance(space, Discrete):
        return x.view(x.shape[0], *space.shape)
    elif isinstance(space, Dict):
        off = 0
        d = {}
        # Assume the last dimension was used for concat
        for key, s in space.spaces.items():
            assert len(s.shape) == 1
            d[key] = x.narrow(-1, off, s.shape[0])
            off += s.shape[0]
        return d
    else:
        raise NotImplementedError(
            f'No unflatten implementation for {type(space)}'
        )


def box_space(shape, low=-np.inf, high=np.inf, dtype=np.float32):
    space = Box(low=low, high=high, shape=shape, dtype=dtype)
    # The PyTorch RNG should be seeded, so use it to seed the space in case
    # we'll draw any samples from it.
    space.seed(th.randint(1 << 31, (1,)).item())
    return space
