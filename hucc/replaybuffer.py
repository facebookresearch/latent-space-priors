# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch as th

log = logging.getLogger(__name__)


@dataclass
class TensorSpec:
    shape: th.Size
    dtype: th.dtype

    @staticmethod
    def fromTensor(x: th.Tensor):
        return TensorSpec(shape=th.Size(x.shape), dtype=x.dtype)


class ReplayBuffer:
    @staticmethod
    def specsFromTensors(tensors: Dict[str, th.Tensor]):
        return {k: TensorSpec.fromTensor(v) for k, v in tensors.items()}

    # If specs is None, do lazy initialization in the first put_row() call.
    def __init__(
        self,
        size: int,
        interleave: int,
        specs: Optional[Dict[str, TensorSpec]] = None,
        device=None,
    ):
        assert size % interleave == 0

        self.max = size
        self.interleave = interleave
        self.start = 0
        self.size = 0
        self.tlen = 0
        self.max_tlen = self.max // self.interleave
        self.device = device

        self._b: Optional[Dict[str, th.Tensor]] = None
        if specs is not None:
            self._init_buffers(specs)

    def _init_buffers(self, specs: Dict[str, TensorSpec]):
        self._b = dict()
        for key, spec in specs.items():
            shape = [self.max] + (
                list(spec.shape) if spec.shape != (1,) else []
            )
            self._b[key] = th.zeros(shape, dtype=spec.dtype, device=self.device)

    def save(self, dest):
        th.save(self.__dict__, dest)

    def load(self, src):
        d = th.load(src)
        for k in self.__dict__.keys():
            self.__dict__[k] = d[k]

    def clear(self) -> None:
        self.start = 0
        self.size = 0
        self.tlen = 0

    def put_row(self, data: Dict[str, th.Tensor]) -> None:
        if self._b is None:  # Lazy auto-initialization
            assert len(data) > 0
            if self.device is None:
                self.device = [v for v in data.values()][0].device
            self._init_buffers(
                self.specsFromTensors({k: v[0] for k, v in data.items()})
            )
        if self._b is None:
            raise RuntimeError()  # to make the linter happy

        bsz = self.interleave
        for k, v in data.items():
            assert v.shape[0] == bsz
        end = (self.start + self.size) % self.max
        idx = th.arange(end, end + bsz) % self.max

        for k in ['obs', 'next_obs']:
            assert (k in data) == (k in self._b), f'No buffer for "{k}"'
            if k in data:
                assert k in self._b
                self._b[k][idx] = data[k].detach().to(self._b[k])

        if 'action' in data:
            assert 'action' in self._b, 'No buffer for "action"'
            self._b['action'][idx] = (
                data['action'].detach().to(self._b['action'])
            )

        for k in set(data.keys()) - set(['obs', 'next_obs', 'action']):
            assert k in self._b, f'No buffer for "{k}"'
            self._b[k][idx] = data[k].squeeze(-1).detach().to(self._b[k])

        if self.size + bsz > self.max:
            self.start = (self.start + bsz) % self.max
            self.size = self.max
        else:
            self.size += bsz
        self.tlen = min(self.tlen + 1, self.max_tlen)

    def put_rows(self, data: Dict[str, th.Tensor]) -> None:
        if self._b is None:  # Lazy auto-initialization
            assert len(data) > 0
            if self.device is None:
                self.device = [v for v in data.values()][0].device
            self._init_buffers(
                self.specsFromTensors({k: v[0] for k, v in data.items()})
            )
        if self._b is None:
            raise RuntimeError()  # to make the linter happy

        # data contains a plain concatenation of several rows
        bsz = self.interleave
        nrows = 0
        for k, v in data.items():
            if nrows == 0:
                assert v.shape[0] % bsz == 0
                nrows = v.shape[0] // bsz
            else:
                assert v.shape[0] == bsz * nrows
        assert bsz * nrows <= self.max
        end = (self.start + self.size) % self.max
        idx = th.arange(end, end + bsz * nrows) % self.max

        for k in ['obs', 'next_obs']:
            assert (k in data) == (k in self._b), f'No buffer for "{k}"'
            if k in data:
                assert k in self._b
                self._b[k][idx] = data[k].detach().to(self._b[k])

        if 'action' in data:
            assert 'action' in self._b, 'No buffer for "action"'
            self._b['action'][idx] = (
                data['action'].detach().to(self._b['action'])
            )

        for k in set(data.keys()) - set(['obs', 'next_obs', 'action']):
            assert k in self._b, f'No buffer for "{k}"'
            self._b[k][idx] = data[k].squeeze(-1).detach().to(self._b[k])

        if self.size + bsz * nrows > self.max:
            self.start = (self.start + bsz * nrows) % self.max
            self.size = self.max
        else:
            self.size += bsz * nrows
        self.tlen = min(self.tlen + nrows, self.max_tlen)

    def get_trajs(
        self, n: int, traj_len: int = 1, priority_key: str = None, device=None
    ) -> Dict[str, th.Tensor]:
        if self._b is None:
            raise RuntimeError(
                'Cannot obtain trajectories from uninitialized ReplayBuffer'
            )
        if device is None:
            device = self.device
        ilv = self.interleave

        if priority_key is not None:
            raise NotImplementedError()
        else:
            # This is with replacement
            idx = (
                th.randint(low=0, high=self.size - ilv * traj_len, size=(n,))
                + self.start
            )

        idx = idx.to(self.device)
        batch: Dict[str, th.Tensor] = dict()
        batch['_idx'] = idx

        for k in set(self._b.keys()):
            s = [
                self._b[k].index_select(0, (idx + i * ilv) % self.max)
                for i in range(traj_len)
            ]
            batch[k] = th.stack(s, dim=1)

        for k, v in batch.items():
            batch[k] = v.to(device)
        return batch

    def pop_row_front(self, device=None):
        if self._b is None:
            raise RuntimeError(
                'Cannot pop items from uninitialized ReplayBuffer'
            )
        if device is None:
            device = self.device
        ilv = self.interleave
        if self.size < ilv:
            raise RuntimeError('ReplayBuffer is empty')

        batch: Dict[str, th.Tensor] = dict()
        for k in set(self._b.keys()):
            batch[k] = self._b[k].narrow(0, self.start, ilv)
        for k, v in batch.items():
            batch[k] = v.to(device)
        self.start = (self.start + ilv) % self.max
        self.size -= ilv
        return batch

    def pop_rows_front(self, n: int, device=None):
        if self._b is None:
            raise RuntimeError(
                'Cannot pop items from uninitialized ReplayBuffer'
            )
        if device is None:
            device = self.device
        ilv = self.interleave
        if self.size < ilv * n:
            raise RuntimeError('ReplayBuffer does not have enough items')

        batch: Dict[str, th.Tensor] = dict()
        for k in set(self._b.keys()):
            batch[k] = self._b[k].narrow(0, self.start, ilv * n)
        for k, v in batch.items():
            batch[k] = v.to(device)
        self.start = (self.start + n * ilv) % self.max
        self.size -= ilv * n
        return batch

    def get_batch_back(self, size: int, device=None, keys=None):
        if self._b is None:
            raise RuntimeError(
                'Cannot query items from uninitialized ReplayBuffer'
            )
        if device is None:
            device = self.device
        if self.size < size:
            raise RuntimeError('ReplayBuffer does not contain enough items')
        if keys is None:
            keys = self._b.keys()

        batch: Dict[str, th.Tensor] = dict()
        idx = th.arange(-size, 0, device=self.device) + self.start + self.size
        idx_mod = idx % self.max
        for k in set(keys):
            batch[k] = self._b[k].index_select(0, idx_mod).to(device)
        return batch

    def get_batch(
        self,
        size: int,
        stack_obs: Optional[int] = None,
        priority_key: str = None,
        device=None,
        out=None,
    ):
        if self._b is None:
            raise RuntimeError(
                'Cannot obtain items from uninitialized ReplayBuffer'
            )
        if device is None:
            device = self.device
        ilv = self.interleave

        def sample_indices():
            if priority_key is not None:
                # XXX hacky, not really working if we really use the
                # circular buffer and pop from the front.
                return th.multinomial(
                    self._b[priority_key][: self.size], size, replacement=True
                ).to(self.device)

            # This is with replacement
            return (
                th.randint(
                    low=0, high=self.size, size=(size,), device=self.device
                )
                + self.start
            )

        # This is with replacement
        idx = sample_indices()
        while stack_obs and stack_obs > 1 and 'terminal' in self._b:
            raise NotImplementedError('Let\'s not use this feature...')
            # Select indices which won't include terminal states in [0,...,stack-1]
            found_term = False
            for i in range(stack_obs):
                if (
                    self._b['terminal']
                    .index_select(0, (idx + i * ilv) % self.max)
                    .any()
                ):
                    log.debug('Found terminal state, resampling')
                    found_term = True
                    break
            if not found_term:
                break
            idx = sample_indices()

        batch: Dict[str, th.Tensor] = out if out is not None else dict()
        if '_idx' in batch:
            batch['_idx'].copy_(idx)
        else:
            batch['_idx'] = idx.to(device)

        if stack_obs:
            if 'obs' in self._b:
                obs = th.stack(
                    [
                        self._b['obs'].index_select(
                            0, (idx + i * ilv) % self.max
                        )
                        for i in range(stack_obs)
                    ],
                    dim=1,
                )
                if 'obs' in batch:
                    batch['obs'].copy_(obs)
                else:
                    batch['obs'] = obs.to(device)
            if 'next_obs' in self._b:
                next_obs = th.stack(
                    [
                        self._b['next_obs'].index_select(
                            0, (idx + (i + 1) * ilv) % self.max
                        )
                        for i in range(stack_obs)
                    ],
                    dim=1,
                )
                if 'next_obs' in batch:
                    batch['next_obs'].copy_(next_obs)
                else:
                    batch['next_obs'] = next_obs.to(device)
            for k in set(self._b.keys()) - set(['obs', 'next_obs']):
                v = self._b[k].index_select(0, (idx + stack_obs - 1) % self.max)
                if k in batch:
                    batch[k].copy_(v)
                else:
                    batch[k] = v.to(device)
        else:
            idx_mod = idx % self.max
            for k in set(self._b.keys()):
                if k in batch:
                    if device == self.device:
                        tptr = batch[k].data_ptr()
                        th.index_select(self._b[k], 0, idx_mod, out=batch[k])
                        assert batch[k].data_ptr() == tptr
                    else:
                        batch[k].copy_(self._b[k].index_select(0, idx_mod))
                else:
                    batch[k] = self._b[k].index_select(0, idx_mod).to(device)

        return batch

    def get_batch_where(
        self,
        size: int,
        indices: th.Tensor,
        stack_obs: Optional[int] = None,
        device=None,
        out=None,
    ):
        if self._b is None:
            raise RuntimeError(
                'Cannot obtain items from uninitialized ReplayBuffer'
            )
        if device is None:
            device = self.device
        ilv = self.interleave

        def sample_indices():
            # This is with replacement
            return indices.view(-1)[
                th.randint(
                    low=0,
                    high=indices.numel(),
                    size=(size,),
                    device=self.device,
                )
            ]

        # This is with replacement
        idx = sample_indices()
        while stack_obs and stack_obs > 1 and 'terminal' in self._b:
            raise NotImplementedError('Let\'s not use this feature...')
            # Select indices which won't include terminal states in [0,...,stack-1]
            found_term = False
            for i in range(stack_obs):
                if (
                    self._b['terminal']
                    .index_select(0, (idx + i * ilv) % self.max)
                    .any()
                ):
                    log.debug('Found terminal state, resampling')
                    found_term = True
                    break
            if not found_term:
                break
            idx = sample_indices()

        batch: Dict[str, th.Tensor] = out if out is not None else dict()
        if '_idx' in batch:
            batch['_idx'].copy_(idx)
        else:
            batch['_idx'] = idx.to(device)

        if stack_obs:
            if 'obs' in self._b:
                obs = th.stack(
                    [
                        self._b['obs'].index_select(
                            0, (idx + i * ilv) % self.max
                        )
                        for i in range(stack_obs)
                    ],
                    dim=1,
                )
                if 'obs' in batch:
                    batch['obs'].copy_(obs)
                else:
                    batch['obs'] = obs.to(device)
            if 'next_obs' in self._b:
                next_obs = th.stack(
                    [
                        self._b['next_obs'].index_select(
                            0, (idx + (i + 1) * ilv) % self.max
                        )
                        for i in range(stack_obs)
                    ],
                    dim=1,
                )
                if 'next_obs' in batch:
                    batch['next_obs'].copy_(next_obs)
                else:
                    batch['next_obs'] = next_obs.to(device)
            for k in set(self._b.keys()) - set(['obs', 'next_obs']):
                v = self._b[k].index_select(0, (idx + stack_obs - 1) % self.max)
                if k in batch:
                    batch[k].copy_(v)
                else:
                    batch[k] = v.to(device)
        else:
            idx_mod = idx % self.max
            for k in set(self._b.keys()):
                if k in batch:
                    if device == self.device:
                        tptr = batch[k].data_ptr()
                        th.index_select(self._b[k], 0, idx_mod, out=batch[k])
                        assert batch[k].data_ptr() == tptr
                    else:
                        batch[k].copy_(self._b[k].index_select(0, idx_mod))
                else:
                    batch[k] = self._b[k].index_select(0, idx_mod).to(device)

        return batch
