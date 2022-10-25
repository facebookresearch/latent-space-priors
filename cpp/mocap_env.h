/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>

#include "common.h"

namespace transforms {
th::Tensor quat_conj(th::Tensor q);
th::Tensor quat_mul(th::Tensor a, th::Tensor b);
th::Tensor quat_diff(th::Tensor a, th::Tensor b);
th::Tensor quat_inv(th::Tensor q);
th::Tensor quat_log(th::Tensor q, double tol = 1e-10);
th::Tensor quat_log2(th::Tensor q, float tol = 1e-7);
th::Tensor quat_dist(th::Tensor a, th::Tensor b);
th::Tensor bounded_quat_dist(th::Tensor a, th::Tensor b);
th::Tensor bounded_quat_dist2(th::Tensor a, th::Tensor b);
th::Tensor quat_to_mat(th::Tensor q);
th::Tensor quat_to_axisangle(th::Tensor q);
} // namespace transforms

enum Observation {
  ActuatorActivation,
  AppendagesPos,
  BodyHeight,
  BodyQuaternions,
  ClipID,
  EndEffectorsPos,
  Frame,
  GyroControl,
  JointsPos,
  JointsVel,
  JointsVelControl,
  QPos,
  QVel,
  Quaternion,
  ReferenceAppendagesPos,
  ReferenceRelRootPosLocalCurrent,
  ReferenceRelRootXYPosCurrent,
  ReferenceRelRootQuatCurrent,
  RefJointsPos,
  RefJointsVel,
  RefAppendages,
  RefVelocity,
  RefAngularVelocity,
  RootAngularVelocity,
  RootVelocity,
  SensorsAccelerometer,
  SensorsForce,
  SensorsGyro,
  SensorsTorque,
  SensorsTouch,
  SensorsVelocimeter,
  Tick,
  TimeInClip,
  VelocimeterControl,
  WorldZAxis,
  CustomStart = 10000,
};

struct Clip {
  TensorMap data;
  TensorMap data32;
  int start_steps = 0; // TODO?
  float dt;
  float duration;

  int num_frames() const;
};

struct ClipData {
  std::vector<std::pair<int, int>> possible_starts;
  th::Tensor start_probabilities;
  std::vector<std::string> clip_ids;
  std::vector<Clip> clips;

  ClipData() {}
};

struct TaskData {
  std::vector<int> ref_steps = {1, 2, 3, 4, 5};
  int max_ref_step;
  double termination_error_threshold;
  int n_sub_steps;
  double control_timestep;
  std::unordered_map<std::string, std::vector<std::string>> observables;
  std::string model_xml;
  th::Tensor body_idxs;
  th::Tensor mocap_to_observable_joint_order;
  th::Tensor ghost_offset;

  TaskData() {}
};

using RewardMap = std::unordered_map<std::string, float>;

class MocapEnv {
public:
  MocapEnv(std::shared_ptr<ClipData> clipd, TaskData const &taskd,
           float time_limit, int reference_dims, int ref_lag,
           std::unordered_map<std::string, std::vector<std::string>>
               observables_override = {},
           bool verbose = false);
  ~MocapEnv();
  std::shared_ptr<ClipData> clipd() { return clipd_; }
  TensorMap action_spec();
  TensorMap observation_spec();
  void seed(int64_t i);
  void set_clip(std::string const &id);
  void set_clip_features(TensorMap const &features);
  void set_early_termination(bool val);
  void set_init_noise(float val);
  void set_init_offset(std::pair<int, int> val);
  void set_end_with_mocap(bool val);
  void set_reward_type(std::string const &val);
  void set_init_state(std::string const &val);
  std::tuple<TensorMap, bool, double, bool, RewardMap> reset();
  std::tuple<TensorMap, bool, double, bool, RewardMap> step(th::Tensor actions);
  th::Tensor render(int width, int height);
  mjData *data() { return d_; }
  mjModel *model() { return m_; }

private:
  void recompile_physics_and_update_observables();
  void after_compile();
  void update_features();
  void save_prev_features();
  void compute_termination_error(mjModel *m, MjDataTensors *dt);
  TensorMap get_all_reference_observations();
  th::Tensor get_reference_rel_bodies_pos_local();
  th::Tensor get_reference_rel_joints();
  th::Tensor get_reference_rel_bodies_pos_global();
  th::Tensor get_reference_ego_bodies_quats();
  th::Tensor get_reference_rel_root_quat();
  th::Tensor get_reference_rel_bodies_quats();
  th::Tensor get_reference_rel_root_pos_local();
  th::Tensor get_single_observation(Observation o);
  TensorMap get_observation();
  void get_clip_to_track();
  RewardMap get_reward(mjModel *m, mjData *d);
  bool should_terminate_episode() const;

private:
  mjModel *m_ = nullptr;
  mjData *d_ = nullptr;
  MjDataTensors dt_;
  std::unique_ptr<MjRenderContext> rctx_;
  std::shared_ptr<ClipData> clipd_;
  std::string model_xml_;
  float time_limit_;
  int n_sub_steps_;
  double control_timestep_;
  std::vector<int> ref_steps_;
  th::Tensor tref_steps_;
  th::Tensor tref_steps_8x_;
  th::Tensor time_steps_;
  th::Tensor time_steps_8x_;
  int max_ref_step_;
  bool reset_next_step_ = true;
  bool raise_exception_on_physics_error_ = true;
  TensorMap reference_observations_;
  std::vector<int> end_effector_geom_ids_;
  std::vector<int> body_geom_ids_;
  std::unordered_map<std::string, std::vector<Observation>> observables_;
  std::unordered_set<std::string> observables_set_;
  std::unordered_map<int, std::string> custom_observables_;
  int clip_idx_;
  int fixed_clip_idx_ = -1;
  TensorMap fixed_clip_features_;
  std::vector<std::unordered_map<int, th::Tensor>> reference_ego_bodies_quats_;
  TensorMap *clip_reference_features_ = nullptr;
  TensorMap current_reference_features_;
  TensorMap walker_features_;
  TensorMap walker_features_prev_;
  int time_step_ = 0;
  int last_step_;
  int tick_ = 0;
  float current_start_time_;
  double termination_error_;
  double termination_error_threshold_;
  bool early_termination_ = true;
  float init_noise_ = 0.0f;
  std::pair<int, int> init_offset_ = {0, 0};
  bool end_mocap_ = false;
  th::Tensor walker_joints_;
  th::Tensor body_idxs_;
  th::Tensor mocap_to_observable_joint_order_;
  th::Tensor xpos_quat_idxs_;
  th::Tensor end_effectors_idx_;
  th::Tensor appendages_idx_;
  bool first_update_features_ = true;
  th::Tensor ghost_offset_;
  int reference_dims_;
  int ref_lag_;
  bool verbose_;
  at::Generator rng_;
  bool end_with_mocap_ = true;
  std::string reward_type_ = "comic";
  std::string init_state_ = "mocap";
  th::Tensor joints_ref_mask_;
};

class BatchedMocapEnv {
public:
  BatchedMocapEnv(std::shared_ptr<ClipData> clipd, TaskData const &taskd,
                  float time_limit, int num_envs, int device,
                  int reference_dims = 1, int ref_lag = 1,
                  bool verbose = false);
  ~BatchedMocapEnv();
  int num_envs() { return envs_.size(); }
  void set_clip(std::string const &id) {
    for (auto &env : envs_) {
      env->set_clip(id);
    }
  }
  void set_clip_features(TensorMap const &features) {
    for (auto &env : envs_) {
      env->set_clip_features(features);
    }
  }
  void set_early_termination(bool val) {
    for (auto &env : envs_) {
      env->set_early_termination(val);
    }
  }
  void set_init_noise(float val) {
    for (auto &env : envs_) {
      env->set_init_noise(val);
    }
  }
  void set_init_offset(std::pair<int, int> val) {
    for (auto &env : envs_) {
      env->set_init_offset(val);
    }
  }
  void set_end_with_mocap(bool val) {
    for (auto &env : envs_) {
      env->set_end_with_mocap(val);
    }
  }
  void set_reward_type(std::string const &val) {
    for (auto &env : envs_) {
      env->set_reward_type(val);
    }
  }
  void set_init_state(std::string const &val) {
    for (auto &env : envs_) {
      env->set_init_state(val);
    }
  }
  void seed(int64_t i) {
    auto j = 0;
    for (auto &env : envs_) {
      env->seed(i + j);
      j++;
    }
  }
  void seeds(std::vector<int64_t> i) {
    if (i.size() == 1) {
      seed(i[0]);
      return;
    } else if (i.size() == envs_.size()) {
      auto j = 0;
      for (auto &env : envs_) {
        env->seed(i[j++]);
      }
    } else {
      throw std::runtime_error("Invalid number of seeds");
    }
  }
  std::vector<mjData *> data() {
    std::vector<mjData *> r;
    for (auto &env : envs_) {
      r.push_back(env->data());
    }
    return r;
  }
  std::vector<mjModel *> model() {
    std::vector<mjModel *> r;
    for (auto &env : envs_) {
      r.push_back(env->model());
    }
    return r;
  }
  std::shared_ptr<ClipData> clipd() { return clipd_; }
  TensorMap action_spec() { return envs_[0]->action_spec(); }
  TensorMap observation_spec();
  TensorMap stack_obs(std::vector<TensorMap> const &obs);
  TensorMap reset();
  TensorMap reset_if_done();
  std::tuple<TensorMap, th::Tensor, th::Tensor, TensorMap>
  step(th::Tensor actions);
  th::Tensor render_single(int width, int height, int idx);

private:
  void run(int idx);

private:
  std::shared_ptr<ClipData> clipd_;
  std::vector<std::unique_ptr<MocapEnv>> envs_;
  std::unordered_map<std::string, std::vector<std::string>> observables_;
  std::unordered_multimap<std::string, th::Tensor>
      last_obs_;            // just views on flattened_obs_ (of single features)
  TensorMap flattened_obs_; // actual returned observation
  th::Tensor rewards_;
  TensorMap reward_terms_;
  th::Tensor done_;
  th::TensorOptions ret_opts_;
  std::vector<std::thread> threads_;
  std::mutex th_mutex_;
  std::condition_variable th_cvar_;
  std::vector<std::string> th_action_;
  th::Tensor actions_;
  Barrier th_barrier_;
  std::map<int, TensorMap> initial_obs_;
  bool verbose_;
};
