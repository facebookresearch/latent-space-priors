# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import bisk
from bisk.helpers import add_robot, root_with_floor
from fairmotion.data import amass


class AMASSReplayEnv(bisk.BiskEnv):
    def __init__(self, render_ghost=True):
        super().__init__()

        root = root_with_floor()
        _, robot_pos = add_robot(root, 'HumanoidAMASSPC', 'robot')
        if render_ghost:
            _, ghost_pos = add_robot(root, 'HumanoidCMUPC', 'ghost')
            root.asset.material['ghost/self'].rgba = [1, 1, 1, 0.5]
            root.find('default', 'ghost/humanoid').geom.set_attributes(
                conaffinity=0, contype=0
            )

        frameskip = 5
        fs = root.find('numeric', 'robot/frameskip')
        if fs is not None:
            frameskip = int(fs.data[0])
        with open('/tmp/out.xml', 'wt') as f:
            f.write(root.to_xml_string())
        self.init_sim(root, frameskip)

        self.init_qpos[0:3] = [0.0, 0.0, 1.3]
        self.init_qpos[3:7] = [0.859, 1.0, 1.0, 0.859]
        if render_ghost:
            self.init_qpos[63 : 63 + 3] = [0.0, 2.0, 1.3]
            self.init_qpos[63 + 3 : 63 + 7] = [0.859, 1.0, 1.0, 0.859]

        self.qpos = None

        self.seed()

    def set_motion(self, qpos, jpos):
        self.qpos = qpos
        self.jpos = jpos
        self.frame = 0

    def get_observation(self):
        return None

    def init_sim(self, root, frameskip):
        for i, j in enumerate(amass.joint_names):
            root.worldbody.add(
                'site',
                name=f'sk_{j}',
                dclass='robot/skeleton_site',
                pos=[i, 0, 1],
            )
        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()
        self.p.data.qpos[:] = self.init_qpos
        self.p.data.qvel[:] = self.init_qvel

    def step(self, action):
        if self.qpos is not None:
            self.p.data.qpos[:] = self.qpos[self.frame]
            for k, v in self.jpos[self.frame].items():
                self.p.named.model.site_pos[f'sk_{k}'] = v
            self.frame = (self.frame + 1) % self.qpos.shape[0]

        try:
            self.p.forward()
        except PhysicsError as e:
            log.exception(e)
            return self.get_observation(), -1, True, {'physics_error': True}
        return self.get_observation(), 0, False, {}
