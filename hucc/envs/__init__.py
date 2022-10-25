# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from functools import partial
from typing import Optional, Union

import gym
from bisk.features import register_featurizer
from gym.envs.registration import register

from hucc.envs.features import (bodyfeet_featurizer, bodyfeet_relz_featurizer,
                                comic_featurizer, comic_proprio_featurizer,
                                comic_proprio_relz_featurizer,
                                comic_propriozd6_featurizer,
                                comic_propriozd6_relz_featurizer,
                                comic_propriozd_featurizer,
                                comic_propriozd_relz_featurizer,
                                comic_relz_featurizer)

log = logging.getLogger(__name__)

register_featurizer('bodyfeet', bodyfeet_featurizer)
register_featurizer('bodyfeet-relz', bodyfeet_relz_featurizer)
register_featurizer('comic', comic_featurizer)
register_featurizer('comic-relz', comic_relz_featurizer)
register_featurizer('comic-proprioceptive', comic_proprio_featurizer)
register_featurizer('comic-proprioceptive-relz', comic_proprio_relz_featurizer)
register_featurizer('comic-proprioceptive-zdelta', comic_propriozd_featurizer)
register_featurizer(
    'comic-proprioceptive-zdelta-relz', comic_propriozd_relz_featurizer
)
register_featurizer('comic-proprioceptive-zdelta6', comic_propriozd6_featurizer)
register_featurizer(
    'comic-proprioceptive-zdelta6-relz', comic_propriozd6_relz_featurizer
)

register(
    id='CMUMocap2020CppMulti-v1',
    entry_point='hucc.envs.mocap_cpp:CMUMocap2020CppMulti',
)
