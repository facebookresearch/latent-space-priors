# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from types import SimpleNamespace

import gym
from omegaconf import DictConfig
from torch import nn

from hucc.agents.agent import Agent
from hucc.agents.guidedsac import GuidedSACAgent
from hucc.agents.guidedsac2 import GuidedSAC2Agent
from hucc.agents.sac import SACAgent
from hucc.agents.sachrl import SACHRLAgent
from hucc.agents.vmpo import VMPOAgent
from hucc.agents.vmpocomic import VMPOCoMicAgent
from hucc.agents.zpriorplan import ZPriorPlanAgent


def agent_cls(name: str):
    return {
        'guidedsac': GuidedSACAgent,
        'guidedsac2': GuidedSAC2Agent,
        'sac': SACAgent,
        'sachrl': SACHRLAgent,
        'vmpo': VMPOAgent,
        'vmpocomic': VMPOCoMicAgent,
        'zpriorplan': ZPriorPlanAgent,
    }[name]


def effective_observation_space(cfg: DictConfig, env: gym.Env):
    return agent_cls(cfg.name).effective_observation_space(env=env, cfg=cfg)


def effective_action_space(cfg: DictConfig, env: gym.Env):
    return agent_cls(cfg.name).effective_action_space(env=env, cfg=cfg)


def make_agent(
    cfg: DictConfig, env: gym.Env, model: nn.Module, optim: SimpleNamespace
) -> Agent:
    return agent_cls(cfg.name)(env=env, model=model, optim=optim, cfg=cfg)
