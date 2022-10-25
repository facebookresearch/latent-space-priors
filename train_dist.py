# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import os
import re
import shutil
import signal
import uuid
from copy import copy
from os import path as osp
from pathlib import Path
from subprocess import Popen

import hydra
import torch as th
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import hucc
from train import TrainingSetup

log = logging.getLogger(__name__)


def jemalloc_path():
    f = os.popen('/sbin/ldconfig -p 2>/dev/null')
    try:
        data = f.read()
    finally:
        f.close()
    expr = r'\s+(lib%s\.[^\s]+)\s+.*=> ([^\s]*)' % (re.escape('jemalloc'))
    res = re.search(expr, data)
    if not res:
        raise RuntimeError('jemalloc not found')
    return res.group(2)


def numa_map():
    f = os.popen('nvidia-smi topo -m 2>/dev/null')
    try:
        data = f.read()
    finally:
        f.close()
    expr = r'^GPU([0-9])(.*)'
    numa_nodes = {}
    for l in data.split('\n'):
        res = re.search(expr, l)
        if res:
            affinity = res.group(2).split('\t')[-1]
            if '-' in affinity:
                affinity = affinity.split('-')[0]
            numa_nodes[int(res.group(1))] = int(affinity)
    # Some machines have a single NUMA node for all GPUs (but still have two
    # NUMA nodes in total) -- in this case, just scrap the assignments
    if len(set(numa_nodes.values())) == 1:
        log.info('All GPUs mapped to a single NUMA node -- skipping assigment')
        return {}
    return numa_nodes


def checkpoint(setup):
    # For check-pointing distributed V-MPO, simply save a JSON file indicating
    # that continuing from the last checkpoint is ok. Since we're on-policy, we
    # don't care about the replay buffer. We might lose a bunch of samples this
    # way, but do not need to modify train.py or agent code.
    cp_path = setup.cfg.checkpoint_path
    if not Path(cp_path).is_file():
        log.info('No checkpoint available, not saving training state')
        return

    log.info('Saving training state to continue from last checkpoint')
    try:
        with open(setup.training_state_path, 'wt') as f:
            json.dump({'continue_from_checkpoint': True}, f)
    except:
        log.exception('Saving training state failed')
    # os.remove(rdvu_file)


def restore(setup):
    # Produce training_state for train.py by parsing the elapsed number of
    # samples from the checkpoint file. XXX this is a hack since we rely on
    # agent internals for checkpointing.
    ts_path = setup.training_state_path
    if Path(ts_path).is_file():
        try:
            with open(ts_path, 'rt') as f:
                d = json.load(f)
            if not d.get('continue_from_checkpoint', False):
                return
        except:
            log.exception('Restoring training state failed')
    else:
        return

    cfg = setup.cfg
    cp_path = cfg.checkpoint_path
    if Path(cp_path).is_file():
        try:
            d = th.load(cp_path)
            n_samples = d['_n_samples']
        except:
            log.exception("Restoring training state failed")
            return
    else:
        return

    try:
        with open(setup.training_state_path, 'wt') as f:
            json.dump({'n_samples': n_samples}, f)
    except:
        log.exception('Restoring training state failed')
        os.remove(setup.training_state_path)
        return

    log.info(f'Restored training state at {n_samples} samples')


# @hydra.main(config_path='config', version_base='1.1')
@hydra.main(config_path='config')
def main(cfg: DictConfig):
    log.info(f'** running from source tree at {hydra.utils.get_original_cwd()}')
    log.info(f'** configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    # Dummy setup for checkpointing
    setup = TrainingSetup(cfg=cfg)
    hucc.set_checkpoint_fn(checkpoint, setup)
    restore(setup)

    env = copy(os.environ)
    procs = []
    jobid = env.get('SLURM_JOBID', uuid.uuid4())
    rscount = env.get('SLURM_RESTART_COUNT', 0)
    rdvu_file = f'{cfg.agent.distributed.rdvu_path}/rdvu-{jobid}-{rscount}'
    setup.rdvu_file = rdvu_file

    args = [
        'python',
        f'{hydra.utils.get_original_cwd()}/train.py',
        '-cp',
        f'{os.getcwd()}/.hydra',
        '-cn',
        'config',
        'hydra.output_subdir=null',
        f'hydra.run.dir={os.getcwd()}',
        f'agent.distributed.rdvu_path={rdvu_file}',
    ]
    ppaths = env.get('PYTHONPATH', '').split(':')
    ppaths.append(hydra.utils.get_original_cwd())
    env['PYTHONPATH'] = ':'.join(ppaths)
    try:
        env['LD_PRELOAD'] = jemalloc_path()
    except:
        log.exception('Failed to use jemalloc')

    # Copy data file to local storage for faster access
    if cfg.get('ref_path', '').startswith('/checkpoint') and cfg.get(
        'copy_refs', True
    ):
        ds_dir = f'/scratch/slurm_tmpdir/{jobid}'
        os.makedirs(ds_dir, exist_ok=True)
        ref_path = f'{ds_dir}/{osp.basename(cfg.ref_path)}'
        log.info(f'Copying reference file {cfg.ref_path} to {ref_path}')
        shutil.copy(cfg.ref_path, ref_path)
        args.append(f'ref_path={ref_path}')

    scaled_args = []
    for arg in cfg.get('scale_by_worldsize', []):
        val = OmegaConf.select(cfg, arg) // cfg.agent.distributed.size
        scaled_args.append(f'{arg}={val}')

    n_nodes = int(env.get('SLURM_NNODES', '1'))
    node_id = int(env.get('SLURM_NODEID', '0'))
    procs_per_node = cfg.agent.distributed.size // n_nodes
    numa_nodes = numa_map()
    procs = []
    for i in range(procs_per_node):
        rank = i + node_id * procs_per_node
        if i in numa_nodes:
            numa_args = [
                'numactl',
                f'--cpunodebind={numa_nodes[i]}',
                f'--membind={numa_nodes[i]}',
            ]
        else:
            numa_args = []
        if 'n_env' in cfg.env.train_args:
            seed = (cfg.seed + rank) * cfg.env.train_args.n_env
        else:
            seed = cfg.seed + rank

        extra_args = [
            f'agent.distributed.rank={rank}',
            f'device=cuda:{i}',
            f'+tb_dir=tb.{rank}',
            f'hydra.job_logging.handlers.file.filename={HydraConfig.get().job.name}-{rank}.log',
            f'max_steps={cfg.max_steps/cfg.agent.distributed.size}',
            f'seed={seed}',
        ]
        extra_args += scaled_args
        if i > 0:
            extra_args += ['eval.interval=1e12']
        log.info(
            f'Launching rank {rank}: {" ".join(numa_args + args + extra_args)}'
        )
        jenv = env
        p = Popen(numa_args + args + extra_args, env=jenv)
        procs.append(p)

    for p in procs:
        p.wait()
    try:
        os.remove(rdvu_file)
    except:
        pass


if __name__ == '__main__':
    main()
