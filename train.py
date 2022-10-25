# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import json
import logging
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import hydra
import numpy as np
import torch as th
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

import hucc
from hucc.agents.utils import discounted_bwd_cumsum_

log = logging.getLogger(__name__)


class TrainingSetup(SimpleNamespace):
    cfg: DictConfig
    agent: hucc.Agent
    model: nn.Module
    tbw: SummaryWriter
    viz: Visdom
    rq: hucc.RenderQueue
    envs: hucc.VecPyTorch
    async_envs: List[hucc.VecPyTorch]
    eval_envs: Dict[str, hucc.VecPyTorch]
    eval_fn: Callable  # Callable[[TrainingSetup, int], None]
    n_samples: int = 0
    replaybuffer_checkpoint_path: str = 'replaybuffer.pt'
    training_state_path: str = 'training_state.json'

    def close(self):
        self.rq.close()
        if self.async_envs:
            for e in self.async_envs:
                e.close()
        else:
            self.envs.close()
        for e in self.eval_envs.values():
            e.close()

        # The replay buffer checkpoint may be huge and we won't need it anymore
        # after training is done.
        try:
            Path(self.replaybuffer_checkpoint_path).unlink()
        except FileNotFoundError:
            pass


def setup_training(cfg: DictConfig) -> TrainingSetup:
    if cfg.device.startswith('cuda') and not th.cuda.is_available():
        log.warning('CUDA not available, falling back to CPU')
        OmegaConf.set_struct(cfg, False)
        cfg.device = 'cpu'
        OmegaConf.set_struct(cfg, True)
    # TODO doesn't work with submitit?
    # if th.backends.cudnn.is_available():
    #    th.backends.cudnn.benchmark = True

    if cfg.get('auto_select_gpu', False) and cfg.device == 'cuda':
        OmegaConf.set_struct(cfg, False)
        cfg.device = (
            f'cuda:{HydraConfig.get().job.num % th.cuda.device_count()}'
        )
        OmegaConf.set_struct(cfg, True)
        log.info(f'Auto-selecting device {cfg.device}')

    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    viz = Visdom(
        server=f'http://{cfg.visdom.host}',
        port=cfg.visdom.port,
        env=cfg.visdom.env,
        offline=cfg.visdom.offline,
        log_to_filename=cfg.visdom.logfile,
    )
    rq = hucc.RenderQueue(viz)

    wrappers = hucc.make_wrappers(cfg.env)
    envs = hucc.make_vec_envs(
        cfg.env.name,
        cfg.env.train_procs,
        raw=cfg.env.raw,
        device=cfg.device,
        seed=cfg.seed,
        wrappers=wrappers,
        **cfg.env.train_args,
    )
    async_envs = []
    if cfg.env.train_instances > 1:
        async_envs.append(envs)
        for i in range(1, cfg.env.train_instances):
            async_envs.append(
                hucc.make_vec_envs(
                    cfg.env.name,
                    cfg.env.train_procs,
                    raw=cfg.env.raw,
                    device=cfg.device,
                    seed=cfg.seed,
                    wrappers=wrappers,
                    **cfg.env.train_args,
                )
            )

    eval_envs = {}
    for k, v in cfg.env.get('eval_instances', {'default': {}}).items():
        args = dict(**cfg.env.eval_args)
        args.update(**v)
        eval_envs[k] = hucc.make_vec_envs(
            cfg.env.name,
            cfg.env.eval_procs,
            raw=cfg.env.raw,
            device=cfg.device,
            seed=cfg.seed,
            wrappers=wrappers,
            **args,
        )

    observation_space = hucc.effective_observation_space(cfg.agent, envs)
    action_space = hucc.effective_action_space(cfg.agent, envs)

    def make_model_rec(mcfg, obs_space, action_space) -> nn.Module:
        if isinstance(obs_space, dict) and isinstance(action_space, dict):
            assert set(obs_space.keys()) == set(action_space.keys())
            models: Dict[str, nn.Module] = {}
            for k in mcfg.keys():
                models[k] = make_model_rec(
                    mcfg[k],
                    obs_space.get(k, envs.observation_space),
                    action_space.get(k, envs.action_space),
                )
            return nn.ModuleDict(models)
        elif isinstance(obs_space, dict):
            models: Dict[str, nn.Module] = {}
            for k in mcfg.keys():
                models[k] = make_model_rec(
                    mcfg[k],
                    obs_space.get(k, envs.observation_space),
                    action_space,
                )
            return nn.ModuleDict(models)
        return hucc.make_model(mcfg, obs_space, action_space)

    model = make_model_rec(cfg.model, observation_space, action_space)
    log.info(f'Model from config:\n{model}')
    model.to(cfg.device)
    optim, sched = hucc.make_optim(cfg.optim, model)

    agent = hucc.make_agent(cfg.agent, envs, model, optim)
    if len(async_envs) > 0 and not agent.supports_async_step:
        raise ValueError(
            f'Selected agent "{cfg.agent.name}" does not support async stepping, can\'t continue with {len(async_envs)} training environment instances.'
        )

    # If the current directoy is different from the original one, assume we have
    # a dedicated job directory. We'll just write our summaries to 'tb/' then.
    try:
        if os.getcwd() != hydra.utils.get_original_cwd():
            tbw = SummaryWriter(cfg.get('tb_dir', 'tb'))
        else:
            tbw = SummaryWriter(cfg.get('tb_dir', ''))
        agent.tbw = tbw
    except:
        # XXX hydra.utils.get_original_cwd throws if we don't run this via
        # run_hydra
        tbw = None

    return TrainingSetup(
        cfg=cfg,
        agent=agent,
        model=model,
        tbw=tbw,
        viz=viz,
        rq=rq,
        envs=envs,
        async_envs=async_envs,
        eval_envs=eval_envs,
        eval_fn=eval,
    )


def eval(setup: TrainingSetup, n_samples: int = -1, eval_seed: int = 0):
    res = []
    for key in setup.eval_envs:
        r = eval_instance(setup, key, n_samples, eval_seed)
        res.append(r)
    return res


def eval_instance(
    setup: TrainingSetup,
    key: str = 'default',
    n_samples: int = -1,
    eval_seed: int = 0,
):
    cfg = setup.cfg
    agent = setup.agent
    rq = setup.rq
    envs = setup.eval_envs[key]

    obs, _ = envs.reset(
        seed=list(range(eval_seed, envs.num_envs + eval_seed))
    )  # Deterministic evals
    reward = th.zeros(envs.num_envs)
    rewards: List[th.Tensor] = []
    dones: List[th.Tensor] = [th.tensor([False] * envs.num_envs)]
    rq_in: List[List[Dict[str, Any]]] = [[] for _ in range(envs.num_envs)]
    n_imgs = 0
    collect_img = cfg.eval.video is not None
    collect_all = collect_img and cfg.eval.video.record_all
    annotate = collect_img and (
        cfg.eval.video.annotations or (cfg.eval.video.annotations is None)
    )
    vmode = cfg.eval.video.get('mode', 'rgb_array') if collect_img else None
    vwidth = int(cfg.eval.video.size[0]) if collect_img else 0
    vheight = int(cfg.eval.video.size[1]) if collect_img else 0
    metrics = set(cfg.eval.metrics.keys())
    metrics_v: Dict[str, Any] = defaultdict(
        lambda: [[] for _ in range(envs.num_envs)]
    )
    extra = None
    entropy_ds = []
    while True:
        if collect_img:
            extra_right: List[List[str]] = [[] for _ in range(envs.num_envs)]
            if extra is not None and isinstance(extra, dict) and 'viz' in extra:
                for i in range(envs.num_envs):
                    for k in extra['viz']:
                        if isinstance(extra[k][i], str):
                            extra_right[i].append(f'{k} {extra[k][i]}')
                        elif isinstance(extra[k][i], np.ndarray):
                            v = np.array2string(
                                extra[k][i], separator=',', precision=2
                            )
                            extra_right[i].append(f'{k} {v}')
                        else:
                            v = np.array2string(
                                extra[k][i].cpu().numpy(),
                                separator=',',
                                precision=2,
                            )
                            extra_right[i].append(f'{k} {v}')
            ekey = 'Eval' if key == 'default' else f'Eval {key}'
            if collect_all:
                for i, img in enumerate(
                    envs.render_all(mode=vmode, width=vwidth, height=vheight)
                ):
                    if dones[-1][i].item():
                        continue
                    rq_in[i].append(
                        {
                            'img': img,
                            's_left': [ekey, f'Samples {n_samples}'],
                            's_right': [
                                f'Trial {i+1}',
                                f'Frame {len(rewards)}',
                                f'Reward {reward[i].item():+.02f}',
                            ]
                            + extra_right[i],
                        }
                    )
            else:
                if not dones[-1][0].item():
                    rq_in[0].append(
                        {
                            'img': envs.render_single(
                                mode=vmode, width=vwidth, height=vheight
                            ),
                            's_left': [ekey, f'Samples {n_samples}'],
                            's_right': [
                                f'Frame {n_imgs}',
                                f'Reward {reward[0].item():+.02f}',
                            ]
                            + extra_right[0],
                        }
                    )
                    n_imgs += 1
                    if n_imgs > cfg.eval.video.length:
                        collect_img = False

        action, extra = agent.action(envs, obs)
        next_obs, reward, term, trunc, info = envs.step(action)
        done = term | trunc
        if 'entropy_d' in envs.ctx:
            entropy_ds.append(envs.ctx['entropy_d'])

        for k in metrics:
            if isinstance(info, list) or isinstance(info, tuple):
                for i in range(len(info)):
                    if dones[-1][i].item():
                        continue
                    if k in info[i]:
                        metrics_v[k][i].append(info[i][k])
        rewards.append(reward.view(-1).to('cpu', copy=True))
        dones.append(done.view(-1).cpu() | dones[-1])
        if dones[-1].all():
            break
        obs, _ = envs.reset_if_done()

    reward = th.stack(rewards, dim=1)
    not_done = th.logical_not(th.stack(dones, dim=1))
    r_undiscounted = (reward * not_done[:, :-1]).sum(dim=1)
    r_discounted = reward.clone()
    discounted_bwd_cumsum_(
        r_discounted, cfg.agent.get('gamma', 1.0), mask=not_done[:, 1:]
    )[:, 0]
    ep_len = not_done.to(th.float32).sum(dim=1)

    metrics_v['episode_length'] = ep_len
    metrics_v['reward'] = th.masked_select(reward, not_done[:, :-1])
    metrics_v['return_disc'] = r_discounted
    metrics_v['return_undisc'] = r_undiscounted
    default_agg = ['mean', 'min', 'max', 'std']
    ekey = 'Eval' if key == 'default' else f'Eval_{key}'
    for k, v in metrics_v.items():
        agg = cfg.eval.metrics.get(k, 'default')
        if isinstance(agg, str):
            if ':' in agg:
                epagg, tagg = agg.split(':')
                if epagg == 'final':
                    v = [ev[-1] for ev in v]
                elif epagg == 'max':
                    v = [max(ev) for ev in v]
                elif epagg == 'min':
                    v = [min(ev) for ev in v]
                elif epagg == 'sum':
                    v = [sum(ev) for ev in v]
                agg = tagg
            elif not isinstance(v, th.Tensor):
                v = itertools.chain(v)
            if agg == 'default':
                agg = default_agg
            else:
                agg = [agg]
        if isinstance(v, th.Tensor):
            agent.tbw_add_scalars(f'{ekey}/{k}', v, agg, n_samples)
        else:
            agent.tbw_add_scalars(
                f'{ekey}/{k}', th.tensor(v).float(), agg, n_samples
            )
    ekey = 'eval' if key == 'default' else f'eval {key}'
    log.info(
        f'{ekey} done, avg len {ep_len.mean().item():.01f}, avg return {r_discounted.mean().item():+.03f}, undisc avg {r_undiscounted.mean():+.03f} min {r_undiscounted.min():+0.3f} max {r_undiscounted.max():+0.3f}'
    )

    if len(entropy_ds) > 0:
        ent_d = (
            th.stack(entropy_ds)
            .T.to(not_done.device)
            .masked_select(not_done[:, :-1])
        )
        agent.tbw_add_scalar('Eval/EntropyDMean', ent_d.mean(), n_samples)
        agent.tbw.add_histogram('Eval/EntropyD', ent_d, n_samples, bins=20)

    if sum([len(q) for q in rq_in]) > 0:
        # Display cumulative reward in video
        c_rew = reward * not_done[:, :-1]
        for i in range(c_rew.shape[1] - 1):
            c_rew[:, i + 1] += c_rew[:, i]
        n_imgs = 0
        if vmode == 'rgb_array':
            for i, ep in enumerate(rq_in):
                for j, input in enumerate(ep):
                    if n_imgs <= cfg.eval.video.length:
                        if annotate:
                            input['s_right'].append(
                                f'Acc. Reward {c_rew[i][j]:+.02f}'
                            )
                            rq.push(**input)
                        else:
                            rq.push(img=input['img'])
                        n_imgs += 1
            rq.plot()
        elif vmode == 'brax_html':
            qps = []
            for i, ep in enumerate(rq_in):
                for j, input in enumerate(ep):
                    if n_imgs <= cfg.eval.video.length:
                        qps.append(input['img'])
            import html

            content = html.escape(envs.html(qps))
            rq.viz.text(
                text=f'<iframe srcdoc="{content}" style="width: {vwidth}px; height: {vheight}px"/>'
            )

    return r_undiscounted.mean().cpu().item()


def train_loop(setup: TrainingSetup):
    cfg = setup.cfg
    agent = setup.agent
    rq = setup.rq
    envs = setup.envs

    agent.train()

    n_envs = envs.num_envs
    cp_path = cfg.checkpoint_path
    record_videos = cfg.video is not None
    annotate = record_videos and (
        cfg.video.annotations or (cfg.video.annotations is None)
    )
    vmode = cfg.video.get('mode', 'rgb_array') if record_videos else 'rgb_array'
    if vmode != 'rgb_array':
        raise NotImplementedError(
            'Video recording in training loop supports rgb_array only'
        )
    vwidth = int(cfg.video.size[0]) if record_videos else 0
    vheight = int(cfg.video.size[1]) if record_videos else 0
    max_steps = int(cfg.max_steps)
    if cfg.get('max_decorrelate_steps', 0) > 0:
        obs = envs.decorrelate(cfg.max_decorrelate_steps)
    else:
        obs, _ = envs.reset(
            seed=list(range(cfg.seed, envs.num_envs + cfg.seed))
        )
    extra = None
    n_imgs = 0
    collect_img = False
    keep_checkpoints = int(cfg.keep_checkpoints)
    agent.train()
    while setup.n_samples < max_steps:
        if setup.n_samples % cfg.eval.interval == 0:
            # Checkpoint time
            try:
                log.debug(
                    f'Checkpointing to {cp_path} after {setup.n_samples} samples'
                )
                with open(f'{cp_path}.tmp', 'wb') as f:
                    agent.save_checkpoint(f)
                os.rename(f'{cp_path}.tmp', cp_path)
                if (
                    keep_checkpoints > 0
                    and setup.n_samples % keep_checkpoints == 0
                ):
                    p = Path(cp_path)
                    cp_unique_path = str(
                        p.with_name(
                            p.stem + f'_{setup.n_samples:08d}' + p.suffix
                        )
                    )
                    shutil.copy(cp_path, cp_unique_path)
            except:
                log.exception('Checkpoint saving failed')

            agent.eval()
            last_eval_ret = setup.eval_fn(setup, setup.n_samples)
            agent.train()

        if record_videos and setup.n_samples % cfg.video.interval == 0:
            collect_img = True
            pass
        if collect_img:
            rqin = {
                'img': envs.render_single(
                    mode='rgb_array', width=vwidth, height=vheight
                )
            }
            if annotate:
                rqin['s_left'] = [
                    f'Samples {setup.n_samples}',
                    f'Frame {n_imgs}',
                ]
                rqin['s_right'] = ['Train']
                if (
                    extra is not None
                    and isinstance(extra, dict)
                    and 'viz' in extra
                ):
                    for k in extra['viz']:
                        if isinstance(extra[k][0], str):
                            rqin['s_right'].append(f'{k} {extra[k][0]}')
                        elif isinstance(extra[k][0], np.ndarray):
                            v = np.array2string(
                                extra[k][0], separator=',', precision=2
                            )
                            rqin['s_right'].append(f'{k} {v}')
                        else:
                            v = np.array2string(
                                extra[k][0].cpu().numpy(),
                                separator=',',
                                precision=2,
                            )
                            rqin['s_right'].append(f'{k} {v}')
            rq.push(**rqin)
            n_imgs += 1
            if n_imgs > cfg.video.length:
                rq.plot()
                n_imgs = 0
                collect_img = False
        action, extra = agent.action(envs, obs)
        next_obs, reward, term, trunc, info = envs.step(action)
        agent.step(
            envs, obs, action, extra, (next_obs, reward, term, trunc, info)
        )
        obs, _ = envs.reset_if_done()
        setup.n_samples += n_envs

    # Final checkpoint & eval time
    try:
        log.debug(f'Checkpointing to {cp_path} after {setup.n_samples} samples')
        with open(f'{cp_path}.tmp', 'wb') as f:
            agent.save_checkpoint(f)
        os.rename(f'{cp_path}.tmp', cp_path)
        if keep_checkpoints > 0 and setup.n_samples % keep_checkpoints == 0:
            p = Path(cp_path)
            cp_unique_path = str(
                p.with_name(p.stem + f'_{setup.n_samples:08d}' + p.suffix)
            )
            shutil.copy(cp_path, cp_unique_path)
    except:
        log.exception('Checkpoint saving failed')

    agent.eval()
    last_eval_ret = setup.eval_fn(setup, setup.n_samples)
    agent.train()

    return last_eval_ret[0]


def train_loop_async(setup: TrainingSetup):
    cfg = setup.cfg
    agent = setup.agent
    rq = setup.rq
    aenvs = setup.async_envs
    executor = ThreadPoolExecutor(len(aenvs) - 1)

    agent.train()

    n_envs = aenvs[0].num_envs  # assumed to be equal for all envs
    cp_path = cfg.checkpoint_path
    record_videos = cfg.video is not None
    annotate = record_videos and (
        cfg.video.annotations or (cfg.video.annotations is None)
    )
    vmode = cfg.video.get('mode', 'rgb_array') if record_videos else 'rgb_array'
    if vmode != 'rgb_array':
        raise NotImplementedError(
            'Video recording in training loop supports rgb_array only'
        )
    vwidth = int(cfg.video.size[0]) if record_videos else 0
    vheight = int(cfg.video.size[1]) if record_videos else 0
    max_steps = int(cfg.max_steps)
    obs = [
        e.reset(seed=list(range(cfg.seed, e.num_envs + cfg.seed)))[0]
        for e in aenvs
    ]

    futures = [None for e in aenvs]

    def lstep(e, a, ex):
        return (a, ex) + e.step(a)

    eidx = 0
    n_imgs = 0
    collect_img = False
    collect_img_eidx = -1
    keep_checkpoints = int(cfg.keep_checkpoints)
    agent.train()
    last_eval = -1
    while setup.n_samples < max_steps:
        if (
            setup.n_samples % cfg.eval.interval == 0
            and setup.n_samples > last_eval
        ):
            last_eval = setup.n_samples
            # Checkpoint time
            try:
                log.debug(
                    f'Checkpointing to {cp_path} after {setup.n_samples} samples'
                )
                with open(cp_path, 'wb') as f:
                    agent.save_checkpoint(f)
                if (
                    keep_checkpoints > 0
                    and setup.n_samples % keep_checkpoints == 0
                ):
                    p = Path(cp_path)
                    cp_unique_path = str(
                        p.with_name(
                            p.stem + f'_{setup.n_samples:08d}' + p.suffix
                        )
                    )
                    shutil.copy(cp_path, cp_unique_path)
            except:
                log.exception('Checkpoint saving failed')

            agent.eval()
            last_eval_ret = setup.eval_fn(setup, setup.n_samples)
            agent.train()

        if record_videos and setup.n_samples % cfg.video.interval == 0:
            collect_img = True
            collect_img_eidx = eidx
            pass
        if collect_img and eidx == collect_img_eidx:
            rqin = {
                'img': aenvs[eidx].render_single(
                    mode='rgb_array', width=vwidth, height=vheight
                )
            }
            if annotate:
                rqin['s_left'] = [
                    f'Samples {setup.n_samples}',
                    f'Frame {n_imgs}',
                ]
                rqin['s_right'] = ['Train']
            rq.push(**rqin)
            n_imgs += 1
            if n_imgs > cfg.video.length:
                rq.plot()
                n_imgs = 0
                collect_img = False

        action, extra = agent.action(aenvs[eidx], obs[eidx])
        futures[eidx] = executor.submit(lstep, aenvs[eidx], action, extra)
        eidx = (eidx + 1) % len(aenvs)
        if futures[eidx] is None:
            continue
        action, extra, next_obs, reward, term, trunc, info = futures[
            eidx
        ].result()
        agent.step(
            aenvs[eidx],
            obs[eidx],
            action,
            extra,
            (next_obs, reward, term, trunc, info),
        )
        obs[eidx], _ = aenvs[eidx].reset_if_done()
        setup.n_samples += n_envs

    # Final checkpoint & eval time
    try:
        log.debug(f'Checkpointing to {cp_path} after {setup.n_samples} samples')
        with open(cp_path, 'wb') as f:
            agent.save_checkpoint(f)
        if keep_checkpoints > 0 and setup.n_samples % keep_checkpoints == 0:
            p = Path(cp_path)
            cp_unique_path = str(
                p.with_name(p.stem + f'_{setup.n_samples:08d}' + p.suffix)
            )
            shutil.copy(cp_path, cp_unique_path)
    except:
        log.exception('Checkpoint saving failed')

    agent.eval()
    last_eval_ret = setup.eval_fn(setup, setup.n_samples)
    agent.train()

    return last_eval_ret[0]


def checkpoint(setup):
    log.info('Checkpointing agent and replay buffer')
    cfg = setup.cfg
    cp_path = cfg.checkpoint_path
    try:
        with open(f'{cp_path}.tmp', 'wb') as f:
            setup.agent.save_checkpoint(f)
        os.rename(f'{cp_path}.tmp', cp_path)
    except:
        log.exception('Checkpointing agent failed')

    if hasattr(setup.agent, '_buffer'):
        try:
            with open(setup.replaybuffer_checkpoint_path, 'wb') as f:
                setup.agent._buffer.save(f)
        except:
            log.exception('Checkpointing replay buffer failed')

    try:
        with open(setup.training_state_path, 'wt') as f:
            json.dump({'n_samples': setup.n_samples}, f)
    except:
        log.exception('Checkpointing training state failed')


def restore(setup):
    ts_path = setup.training_state_path
    if Path(ts_path).is_file():
        try:
            with open(ts_path, 'rt') as f:
                d = json.load(f)
            setup.n_samples = d['n_samples']
        except:
            log.exception('Restoring training state failed')
    else:
        return

    cfg = setup.cfg
    cp_path = cfg.checkpoint_path
    if cp_path and Path(cp_path).is_file():
        log.info(f'Loading agent from checkpoint {cp_path}')
        with open(cp_path, 'rb') as fd:
            setup.agent.load_checkpoint(fd)
    else:
        raise RuntimeError('Found training state but no agent checkpoint')

    rpbuf_path = setup.replaybuffer_checkpoint_path
    if hasattr(setup.agent, '_buffer') and Path(rpbuf_path).is_file():
        try:
            with open(rpbuf_path, 'rb') as f:
                setup.agent._buffer.load(f)
        except:
            log.exception('Restoring replay buffer failed')


def auto_adapt_config(cfg: DictConfig) -> DictConfig:
    if cfg.env.name.startswith('BiskStairs'):
        # Goal space should be postfixed with '-relz' since Z features reported
        # by this environment are wrt to the current geom under the robot
        if 'goal_space' in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg.goal_space = f'{cfg.goal_space}-relz'
            OmegaConf.set_struct(cfg, True)
        if 'comic_obs_lo' in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg.comic_obs_lo = f'{cfg.comic_obs_lo}-relz'
            OmegaConf.set_struct(cfg, True)
        if 'features_lo' in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg.features_lo = f'{cfg.features_lo}-relz'
            OmegaConf.set_struct(cfg, True)
    elif cfg.env.name.startswith('BiskPoleBalance'):
        # High-level acting at every time-step
        if 'action_interval' in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg.action_interval = 1
            OmegaConf.set_struct(cfg, True)
    return cfg


def main_(cfg: DictConfig):
    log.info(f'** running from source tree at {hydra.utils.get_original_cwd()}')
    if cfg.auto_adapt:
        cfg = auto_adapt_config(cfg)
    log.info(
        f'** adapted configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}'
    )

    setup = setup_training(cfg)
    if cfg.init_from:
        log.info(f'Initializing agent from checkpoint {cfg.init_from}')
        with open(cfg.init_from, 'rb') as fd:
            setup.agent.load_checkpoint(fd)
    hucc.set_checkpoint_fn(checkpoint, setup)
    restore(setup)

    if setup.async_envs:
        ret = train_loop_async(setup)
    else:
        ret = train_loop(setup)
    setup.close()

    return ret


@hydra.main(config_path='config')  # , version_base='1.1')
def main(cfg: DictConfig):
    try:
        return main_(cfg)
    except:
        log.exception(f'!!! Unhandled exception:')
        raise


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error(
        'Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback)
    )

    # Enter post-portem debugger unless we're running on slurm
    if os.environ.get('SLURM_JOBID', None):
        return

    import pdb

    pdb.post_mortem(exc_traceback)


if __name__ == '__main__':
    import sys

    sys.excepthook = handle_exception
    main()
