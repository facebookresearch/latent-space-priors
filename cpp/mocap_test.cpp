/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include <pybind11/embed.h> // everything needed for embedding

#include "common_py.h"
#include "mocap_env.h"

namespace py = pybind11;
using namespace pybind11::literals;

int main(int argc, char **argv) {
  char const *homedir;
  if ((homedir = getenv("HOME")) == NULL) {
    homedir = getpwuid(getuid())->pw_dir;
  }
  auto mjkey = std::string(homedir) + "/.mujoco/mjkey.txt";
  std::cout << "Activating with " << mjkey << std::endl;
  mj_activate(mjkey.c_str());

  auto clipd = std::make_shared<ClipData>();
  TaskData taskd;
  {
    py::scoped_interpreter guard{};

    auto locals = py::dict();
    py::exec(R"(
print("hey there!");
import dm_env
print(dm_env.__path__)
from dm_control.locomotion import arenas
print(arenas.__path__)
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.walkers import cmu_humanoid
print("imports done!")

dataset = 'walk_tiny'
n_ref = 5
min_steps = 10
reward_type = 'comic'
walker_type = cmu_humanoid.CMUHumanoidPositionControlledV2020
arena = arenas.Floor()
task = tracking.MultiClipMocapTracking(
    walker=walker_type,
    arena=arena,
    ref_path=cmu_mocap_data.get_path_for_cmu(version='2020'),
    dataset=dataset,
    ref_steps=list(range(1, n_ref+1)),
    min_steps=min_steps,
    reward_type=reward_type,
)
  )",
             py::globals(), locals);

    // Standard features for COMIC
    std::unordered_map<std::string, std::vector<std::string>> observables;
    observables["observation"] = {
        "appendages_pos",      "body_height",           "joints_pos",
        "joints_vel",          "gyro_control",          "joints_vel_control",
        "velocimeter_control", "sensors_touch",         "sensors_velocimeter",
        "sensors_gyro",        "sensors_accelerometer", "end_effectors_pos",
        "actuator_activation", "sensors_torque",        "world_zaxis",
    };
    // observables["time"] = {"time_in_clip"};
    // observables["clip_id"] = {"clip_id"};
    /*
    observables["reference"] = {
        "reference_rel_joints",           "reference_rel_bodies_pos_global",
        "reference_rel_root_pos_local",   "reference_rel_root_quat",
        "reference_rel_bodies_quats",     "reference_appendages_pos",
        "reference_rel_bodies_pos_local", "reference_ego_bodies_quats",
    };
    */
    //    observables["observation"] = {"qpos", "qvel"};

    auto task = locals["task"];
    parseClipData(*clipd, task, observables);
    parseTaskData(taskd, task, observables);
    std::cout << "Parsed all the data" << std::endl;
  }

#if 0
  auto time_limit = 30;
  auto env = MocapEnv(clipd, taskd, time_limit);
  auto obs = env.reset();
  auto action = th::zeros({56});
  auto start = std::chrono::steady_clock::now();
  for (auto i = 0; i < 1000; i++) {
    env.step(action);
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << "1000 steps single in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "[ms]" << std::endl;
#endif
  for (auto n : {1, 2, 5, 10, 20, 40}) {
    auto time_limit = 30;
    BatchedMocapEnv benv(clipd, taskd, time_limit, n, 0);
    benv.set_early_termination(false);
    benv.seed(0);
    benv.reset();
    auto action = th::zeros({n, 56}) + 0.5;
    auto start = std::chrono::steady_clock::now();
    auto steps = 2000;
    for (auto i = 0; i < steps; i++) {
      benv.step(action);
      benv.reset_if_done();
    }
    auto end = std::chrono::steady_clock::now();
    auto throughput =
        double(steps * n) /
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << n << " envs " << std::fixed << std::setprecision(1)
              << throughput * 1000 << " samples/s" << std::endl;
#if 0
    std::cout << "Press enter to start loop..." << std::endl;
    std::cin.get();
    for (auto i = 0; i < 10; i++) {
      benv.step(action);
      benv.reset_if_done();
    }
#endif
  }
  return 0;
}
