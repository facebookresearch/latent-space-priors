# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import logging
import re
import signal
from collections import defaultdict
from copy import deepcopy
from types import FrameType, SimpleNamespace
from typing import Any, Dict

import hydra
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer, lr_scheduler

log = logging.getLogger(__name__)


def make_optim(optim_cfg: DictConfig, model: nn.Module) -> SimpleNamespace:
    def make(ocfg: DictConfig, model: nn.Module):
        cfg = deepcopy(ocfg)
        params = list(model.parameters())
        if len(params) == 0:
            log.info(
                f'Empty parameter list for module, ignoring optimizer settings'
            )
            return None
        device = params[0].device
        if ocfg.get('fuse', False):
            OmegaConf.set_struct(cfg, False)
            if device.type == 'cuda' and ocfg['_target_'] == 'torch.optim.Adam':
                try:
                    from apex.optimizers import FusedAdam
                except ImportError:
                    pass
                else:
                    # TODO This should support zero_grad(set_to_none)
                    ocfg['_target_'] = 'apex.optimizers.FusedAdam'
                    log.info('Using apex.optimizers.FusedAdam')
            del cfg['fuse']
        warmup = ocfg.get('warmup', None)
        if warmup:
            del cfg['warmup']
        linear_decay = ocfg.get('linear_decay', None)
        if linear_decay:
            del cfg['linear_decay']

        optim = hydra.utils.instantiate(cfg, model.parameters())

        sched: Any = None
        if warmup:
            assert not linear_decay
            sched = lr_scheduler.LinearLR(
                optim, start_factor=1e-8, total_iters=100
            )
        elif linear_decay:
            sched = lr_scheduler.LinearLR(
                optim,
                start_factor=1,
                end_factor=float(linear_decay.end_factor),
                total_iters=int(linear_decay.iters),
            )
        else:
            sched = lr_scheduler.ConstantLR(optim, factor=1)

        return optim, sched

    def recurse(ocfg: DictConfig, model: nn.Module):
        if ocfg is None:
            return SimpleNamespace(), SimpleNamespace()

        optims: Dict[str, Optimizer] = {}
        scheds: Dict[str, Optimizer] = {}
        for k, v in ocfg.items():
            if '_target_' in v:
                if k == '_all_':
                    optims[k], scheds[k] = make(v, model)
                else:
                    optims[k], scheds[k] = make(v, getattr(model, k))
            else:
                optims[k], scheds[k] = recurse(v, getattr(model, k))
        return SimpleNamespace(**optims), SimpleNamespace(**scheds)

    return recurse(optim_cfg, model)


def set_checkpoint_fn(fn, *args, **kwargs):
    prev_usr1 = None

    def sigusr1(signum: signal.Signals, frame: FrameType):
        log.info('SIGUSR1 intercepted, calling checkpoint handler')
        fn(*args, **kwargs)
        if (
            prev_usr1 is not None
            and prev_usr1 != signal.SIG_IGN
            and prev_usr1 != signal.SIG_DFL
        ):
            prev_usr1(signum, frame)

    prev_usr1 = signal.signal(signal.SIGUSR1, sigusr1)


# A shorthand for specific gather calls:
# Your input tensor has some extra dimenions that you want to retain. For
# example, input is a BxTxN tensor and you want to select specific time-steps
# via an Bx1 tensor. The only way is to gather(), and that requires you to
# expand the index tensor. Or, `dim_select(input, 1, index)`, and you'll get an
# Bx1xN tensor in return. If the index is just a B tensor, the result will be a
# BxN tensor.
def dim_select(input: th.Tensor, dim: int, index: th.Tensor):
    # TODO this is a bunch of special cases for now... figure out how to
    # generalize it?
    if input.ndim == 2 and index.ndim == 1 and dim == 1:
        return input.gather(1, index.view(-1, 1)).squeeze(1)
    elif input.ndim == 3 and index.ndim == 1 and dim == 0:
        index = index.view(1, -1, 1).expand(1, index.shape[0], input.shape[2])
        return input.gather(0, index).view(-1, input.shape[-1])
    elif input.ndim == 3 and index.ndim == 1 and dim == 1:
        index = index.view(-1, 1, 1).expand(index.shape[0], 1, input.shape[-1])
        return input.gather(1, index).view(-1, input.shape[-1])
    elif input.ndim == 3 and index.ndim == 2 and dim == 1:
        index = index.unsqueeze(-1).expand(*index.shape, input.shape[-1])
        return input.gather(1, index)
    else:
        raise ValueError('Can\'t dim_select this combination of tensors')


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect;
    from https://stackoverflow.com/a/2669120."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def sorted_nicely_sep(l, sep=','):
    by_rank = defaultdict(list)
    for a in l:
        by_rank[len(a.split(sep))].append(a)
    ta_sorted = []
    for k in sorted(by_rank.keys()):
        ta_sorted += sorted_nicely(by_rank[k])
    return ta_sorted


# Context manager to perform common save/restore ops for graph capturing
class capture_graph_context:
    def __init__(self, *objects_to_restore):
        self.objs = objects_to_restore

    def __enter__(self):
        from torch import distributions as D

        import hucc.models.blocks

        self.rngs = {'cpu': th.get_rng_state().clone()}
        if th.cuda.is_available():
            self.rngs['cuda'] = th.cuda.get_rng_state().clone()

        data = []
        for o in self.objs:
            try:
                data.append(o.state_dict())
            except AttributeError:
                data.append(o.clone())
        self.buffer = io.BytesIO()
        th.save(data, self.buffer)

        hucc.models.blocks._GRAPH_CAPTURING = True
        self.distribution_validate_args = D.Distribution._validate_args
        D.Distribution.set_default_validate_args(False)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        from torch import distributions as D

        import hucc.models.blocks

        D.Distribution.set_default_validate_args(
            self.distribution_validate_args
        )
        hucc.models.blocks._GRAPH_CAPTURING = False

        self.buffer.seek(0)
        data = th.load(self.buffer)
        for o, d in zip(self.objs, data):
            try:
                o.load_state_dict(d)
            except AttributeError:
                o.copy_(d)

        th.set_rng_state(self.rngs['cpu'])
        if 'cuda' in self.rngs:
            th.cuda.set_rng_state(self.rngs['cuda'])


def capture_graph(callable, *objects_to_restore, **capture_kwargs):
    with capture_graph_context(*objects_to_restore):
        # Warmup
        th.cuda.synchronize()
        s = th.cuda.Stream()
        s.wait_stream(th.cuda.current_stream())
        with th.cuda.stream(s):
            for _ in range(3):
                callable()
        th.cuda.current_stream().wait_stream(s)

        # Capture
        graph = th.cuda.CUDAGraph()
        with th.cuda.graph(graph, **capture_kwargs):
            res = callable()

    return graph, res
