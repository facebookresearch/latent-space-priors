/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>

#ifdef __unix__
#include <numa.h>
#include <sched.h>
#endif // __unix__
#include <sys/types.h>

#include <mujoco.h>
#include <torch/torch.h>

#include "mocap_env.h"

using namespace torch::indexing;

namespace {

template <class T>
constexpr T const &clamp(T const &v, T const &lo, T const &hi) {
  return v < lo ? lo : hi < v ? hi : v;
}

Observation map_obs_str(std::string const &obs) {
  if (obs == "act" || obs == "actuator_activation") {
    return Observation::ActuatorActivation;
  } else if (obs == "appendages_pos" || obs == "appendages") {
    return Observation::AppendagesPos;
  } else if (obs == "body_height") {
    return Observation::BodyHeight;
  } else if (obs == "body_quaternions") {
    return Observation::BodyQuaternions;
  } else if (obs == "clip_id") {
    return Observation::ClipID;
  } else if (obs == "end_effectors_pos") {
    return Observation::EndEffectorsPos;
  } else if (obs == "frame") {
    return Observation::Frame;
  } else if (obs == "gyro_control") {
    return Observation::GyroControl;
  } else if (obs == "joints_pos" || obs == "joints") {
    return Observation::JointsPos;
  } else if (obs == "joints_vel" || obs == "joints_velocity") {
    return Observation::JointsVel;
  } else if (obs == "joints_vel_control") {
    return Observation::JointsVelControl;
  } else if (obs == "qpos") {
    return Observation::QPos;
  } else if (obs == "qvel") {
    return Observation::QVel;
  } else if (obs == "quaternion") {
    return Observation::Quaternion;
  } else if (obs == "reference_appendages_pos") {
    return Observation::ReferenceAppendagesPos;
  } else if (obs == "reference_rel_root_pos_local_current") {
    return Observation::ReferenceRelRootPosLocalCurrent;
  } else if (obs == "reference_rel_root_xypos_current") {
    return Observation::ReferenceRelRootXYPosCurrent;
  } else if (obs == "reference_rel_root_quat_current") {
    return Observation::ReferenceRelRootQuatCurrent;
  } else if (obs == "ref_joints") {
    return Observation::RefJointsPos;
  } else if (obs == "ref_joints_velocity") {
    return Observation::RefJointsVel;
  } else if (obs == "ref_appendages") {
    return Observation::RefAppendages;
  } else if (obs == "ref_velocity") {
    return Observation::RefVelocity;
  } else if (obs == "ref_angular_velocity") {
    return Observation::RefAngularVelocity;
  } else if (obs == "root_angular_velocity" || obs == "angular_velocity") {
    return Observation::RootAngularVelocity;
  } else if (obs == "root_velocity" || obs == "velocity") {
    return Observation::RootVelocity;
  } else if (obs == "sensors_accelerometer") {
    return Observation::SensorsAccelerometer;
  } else if (obs == "sensors_force") {
    return Observation::SensorsForce;
  } else if (obs == "sensors_gyro") {
    return Observation::SensorsGyro;
  } else if (obs == "sensors_torque") {
    return Observation::SensorsTorque;
  } else if (obs == "sensors_touch") {
    return Observation::SensorsTouch;
  } else if (obs == "sensors_velocimeter") {
    return Observation::SensorsVelocimeter;
  } else if (obs == "tick") {
    return Observation::Tick;
  } else if (obs == "time_in_clip") {
    return Observation::TimeInClip;
  } else if (obs == "velocimeter_control") {
    return Observation::VelocimeterControl;
  } else if (obs == "world_zaxis") {
    return Observation::WorldZAxis;
  }
  return Observation::CustomStart;
}

} // namespace

namespace transforms {

// Ports from MuJoCo so that we can work with floats and doubles

// convert quaternion to 3D rotation matrix
void c_mju_quat2Mat(double res[9], const double quat[4]) {
  // null quat: identity
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 1;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 1;
  }

  // regular processing
  else {
    const double q00 = quat[0] * quat[0];
    const double q01 = quat[0] * quat[1];
    const double q02 = quat[0] * quat[2];
    const double q03 = quat[0] * quat[3];
    const double q11 = quat[1] * quat[1];
    const double q12 = quat[1] * quat[2];
    const double q13 = quat[1] * quat[3];
    const double q22 = quat[2] * quat[2];
    const double q23 = quat[2] * quat[3];
    const double q33 = quat[3] * quat[3];

    res[0] = q00 + q11 - q22 - q33;
    res[4] = q00 - q11 + q22 - q33;
    res[8] = q00 - q11 - q22 + q33;

    res[1] = 2 * (q12 - q03);
    res[2] = 2 * (q13 + q02);
    res[3] = 2 * (q12 + q03);
    res[5] = 2 * (q23 - q01);
    res[6] = 2 * (q13 - q02);
    res[7] = 2 * (q23 + q01);
  }
}

void c_mju_quat2Mat32(float res[9], const float quat[4]) {
  // null quat: identity
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 1;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 1;
  }

  // regular processing
  else {
    const float q00 = quat[0] * quat[0];
    const float q01 = quat[0] * quat[1];
    const float q02 = quat[0] * quat[2];
    const float q03 = quat[0] * quat[3];
    const float q11 = quat[1] * quat[1];
    const float q12 = quat[1] * quat[2];
    const float q13 = quat[1] * quat[3];
    const float q22 = quat[2] * quat[2];
    const float q23 = quat[2] * quat[3];
    const float q33 = quat[3] * quat[3];

    res[0] = q00 + q11 - q22 - q33;
    res[4] = q00 - q11 + q22 - q33;
    res[8] = q00 - q11 - q22 + q33;

    res[1] = 2 * (q12 - q03);
    res[2] = 2 * (q13 + q02);
    res[3] = 2 * (q12 + q03);
    res[5] = 2 * (q23 - q01);
    res[6] = 2 * (q13 - q02);
    res[7] = 2 * (q23 + q01);
  }
}

double constexpr _TOL = 1e-10;

// TODO unit-test those against the dm_control ones?
th::Tensor quat_conj(th::Tensor q) {
  auto static mul =
      th::tensor({1.0, -1.0, -1.0, -1.0},
                 th::TensorOptions().dtype(th::kFloat32).device(th::kCPU));
  return q * mul.to(q.device(), q.dtype());
}

th::Tensor quat_conj2(th::Tensor q) {
  auto static mul =
      th::tensor({1.0, -1.0, -1.0, -1.0},
                 th::TensorOptions().dtype(th::kFloat32).device(q.device()));
  return q * mul;
}

th::Tensor quat_mul(th::Tensor a, th::Tensor b) {
  auto static idx =
      th::tensor({0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0});
  auto static tsign = th::tensor(
      {1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1}, a.device());
  auto qmat = a.index({"...", idx}).mul_(tsign);
  auto ret = qmat.view({-1, 4, 4}).matmul(b.unsqueeze(-1));
  ret.squeeze_(-1);
  return ret;
}

th::Tensor quat_mul2(th::Tensor a, th::Tensor b) {
  auto static idx =
      th::tensor({0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0});
  auto static tsign = th::tensor(
      {1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1}, a.device());
  auto qmat = a.index({"...", idx}).mul_(tsign);
  auto ret = qmat.view({-1, 4, 4}).matmul(b.unsqueeze(-1));
  return ret.squeeze(-1);
}

th::Tensor quat_diff(th::Tensor a, th::Tensor b) {
  auto static idx =
      th::tensor({0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0});
  auto static tsign =
      th::tensor({1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1});
  auto qmat = a.index({"...", idx}).mul_(tsign);
  if (a.dim() == 2) {
    auto ret = qmat.view({-1, 4, 4}).matmul(b.unsqueeze(-1));
    ret.squeeze_(-1);
    return ret;
  } else {
    auto ret = qmat.view({4, 4}).matmul(b.unsqueeze(-1));
    ret.squeeze_(-1);
    return ret;
  }
}

th::Tensor quat_inv(th::Tensor q) {
  auto r = quat_conj(q);
  r.div_((q * q).sum(-1, true));
  return r;
}

th::Tensor quat_inv2(th::Tensor q) {
  auto r = quat_conj2(q);
  r = r.div((q * q).sum(-1, true));
  return r;
}

th::Tensor quat_log(th::Tensor q, double tol) {
  auto q_norm = (q + tol).norm(2, -1, true);
  auto a = q.index({"...", Slice(0, 1)});
  auto v = q.index({"...", Slice(1, 4)}).clone();
  v.div_((v + tol).norm(2, -1, true));
  v.mul_(a.add_(-tol).clip_(-1.0, 1.0).arccos_());
  v.div_(q_norm);
  return th::cat({q_norm.select(-1, 0).log_().unsqueeze_(1), v}, -1);
}

th::Tensor quat_log2(th::Tensor q, float tol) {
  auto q_norm = (q + tol).norm(2, -1, true);
  auto a = q.index({"...", Slice(0, 1)});
  auto v = q.index({"...", Slice(1, 4)}).clone();
  v = v.div((v + tol).norm(2, -1, true));
  v = v.mul(a.add(-tol).clip(-1.0 + tol, 1.0 - tol).arccos());
  v = v.div(q_norm);
  return th::cat({q_norm.select(-1, 0).log().unsqueeze(1), v}, -1);
}

th::Tensor quat_dist(th::Tensor a, th::Tensor b) {
  auto prod = quat_mul(a, quat_inv(b));
  prod.div_(prod.norm(2, -1, true));
  return quat_log(prod).norm(2, -1, true);
}

th::Tensor quat_dist2(th::Tensor a, th::Tensor b) {
  auto prod = quat_mul2(a, quat_inv2(b));
  prod = prod.div(prod.norm(2, -1, true));
  return quat_log2(prod).norm(2, -1, true);
}

th::Tensor bounded_quat_dist(th::Tensor a, th::Tensor b) {
  a = a / a.norm(2, -1, true);
  b = b / b.norm(2, -1, true);
  auto ddist = quat_dist(a, b);
  auto adist = quat_dist(a, -b);
  return th::min(ddist, adist);
}

th::Tensor bounded_quat_dist2(th::Tensor a, th::Tensor b) {
  a = a / a.norm(2, -1, true);
  b = b / b.norm(2, -1, true);
  auto ddist = quat_dist2(a, b);
  auto adist = quat_dist2(a, -b);
  return th::min(ddist, adist);
}

th::Tensor quat_to_mat(th::Tensor q) {
  auto mat = th::zeros({3, 3}, th::kFloat64);
  auto qa = q.accessor<double, 1>();
  double d[4] = {qa[0], qa[1], qa[2], qa[3]};
  c_mju_quat2Mat(static_cast<double *>(mat.data_ptr()), d);
  return mat;
}

th::Tensor quat_to_mat32(th::Tensor q) {
  auto mat = th::zeros({3, 3}, th::kFloat32);
  auto qa = q.accessor<float, 1>();
  float d[4] = {qa[0], qa[1], qa[2], qa[3]};
  c_mju_quat2Mat32(static_cast<float *>(mat.data_ptr()), d);
  return mat;
}

th::Tensor quat_to_axisangle(th::Tensor q) {
  auto angle = 2 * std::acos(clamp(q[0].item<float>(), -1.0f, 1.0f));
  if (angle < _TOL) {
    return th::zeros(3, q.dtype());
  }
  auto qn = std::sin(angle / 2);
  angle = std::fmod((angle + M_PI), (2 * M_PI)) - M_PI;
  auto axis = q.index({Slice(1, 4)}) / qn;
  return axis * angle;
}

} // namespace transforms

using namespace transforms;

int Clip::num_frames() const { return data.at("joints").sizes()[0]; }

MocapEnv::MocapEnv(std::shared_ptr<ClipData> clipd, TaskData const &taskd,
                   float time_limit, int reference_dims, int ref_lag,
                   std::unordered_map<std::string, std::vector<std::string>>
                       observables_override,
                   bool verbose)
    : clipd_(std::move(clipd)), model_xml_(taskd.model_xml),
      time_limit_(time_limit), n_sub_steps_(taskd.n_sub_steps),
      control_timestep_(taskd.control_timestep), ref_steps_(taskd.ref_steps),
      max_ref_step_(taskd.max_ref_step),
      termination_error_threshold_(taskd.termination_error_threshold),
      body_idxs_(taskd.body_idxs.clone()),
      mocap_to_observable_joint_order_(
          taskd.mocap_to_observable_joint_order.clone()),
      ghost_offset_(taskd.ghost_offset), reference_dims_(reference_dims),
      ref_lag_(ref_lag), verbose_(verbose),
      rng_(at::detail::createCPUGenerator()) {
  tref_steps_ = th::tensor(ref_steps_);
  tref_steps_8x_ =
      ((tref_steps_ * 8).repeat({8, 1}) + th::arange(8).view({8, 1}))
          .t()
          .reshape({-1});
  tref_steps_8x_ = tref_steps_.repeat({8});
  time_steps_ = th::zeros_like(tref_steps_);
  time_steps_8x_ = time_steps_.repeat({8});
  xpos_quat_idxs_ = th::tensor({3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                25, 26, 27, 28, 29, 30, 31, 32, 33});

  recompile_physics_and_update_observables();

  std::unordered_map<std::string, int> body_idx;
  for (auto i = 0; i < m_->nbody; i++) {
    auto name = std::string(m_->names + m_->name_bodyadr[i]);
    body_idx[name] = i;
  }
  end_effectors_idx_ = th::tensor({
      body_idx.at("walker/rradius"),
      body_idx.at("walker/lradius"),
      body_idx.at("walker/rfoot"),
      body_idx.at("walker/lfoot"),
  });
  appendages_idx_ = th::tensor({
      body_idx.at("walker/rradius"),
      body_idx.at("walker/lradius"),
      body_idx.at("walker/rfoot"),
      body_idx.at("walker/lfoot"),
      body_idx.at("walker/head"),
  });

  // Some joints are not set in the reference data -- ignore them when
  // computing rewards
  joints_ref_mask_ = th::ones({56}, th::kBool);
  joints_ref_mask_.index_fill_(
      0, th::tensor({6, 13, 29, 30, 31, 41, 42, 43, 53, 54, 55}, th::kInt64),
      0);

  auto observables = taskd.observables;
  if (observables_override.size() > 0) {
    observables = observables_override;
  }
  int custom_obs = 0;
  for (auto const &[k, v] : observables) {
    std::vector<Observation> obss;
    for (auto const &obs : v) {
      int obs_key = map_obs_str(obs);
      if (obs_key == Observation::CustomStart) {
        obs_key += custom_obs;
        custom_obs++;
        custom_observables_[obs_key] = obs;
      }
      obss.push_back((Observation)obs_key);

      observables_set_.insert(obs);
    }
    observables_[k] = obss;
  }

  reference_ego_bodies_quats_.resize(clipd_->clips.size());

  // Init with "dummy" trajectory
  clip_idx_ = 0;
  clip_reference_features_ = &clipd_->clips[clip_idx_].data32;
  walker_joints_ = clip_reference_features_->at("joints")[0];
  for (auto const &it : *clip_reference_features_) {
    walker_features_[it.first] = it.second[0];
  }

  after_compile();
}

MocapEnv::~MocapEnv() {
  if (d_) {
    mj_deleteData(d_);
  }
  if (m_) {
    mj_deleteModel(m_);
  }
}

TensorMap MocapEnv::action_spec() {
  auto ret = TensorMap();
  auto limits = th::from_blob(m_->actuator_ctrlrange, {m_->nu, 2},
                              th::dtype(th::kFloat64));
  ret["minimum"] = limits.index({Slice(), 0}).clone();
  ret["maximum"] = limits.index({Slice(), 1}).clone();
  return ret;
}

TensorMap MocapEnv::observation_spec() {
  // XXX da hacks
  auto obs = get_observation();
  auto ret = TensorMap();
  for (auto const &it : obs) {
    ret[it.first] = th::zeros_like(it.second);
  }
  return ret;
}

void MocapEnv::seed(int64_t s) { rng_.set_current_seed(s); }

void MocapEnv::set_clip(std::string const &id) {
  reset_next_step_ = true;
  clip_reference_features_ = nullptr;

  fixed_clip_features_.clear();
  if (id.empty()) {
    fixed_clip_idx_ = -1;
  }
  auto &ids = clipd_->clip_ids;
  auto it = std::find(ids.cbegin(), ids.cend(), id);
  if (it == ids.cend()) {
    throw std::runtime_error("No such clip in collection: " + id);
  }
  fixed_clip_idx_ = it - ids.cbegin();
  //  std::cout << "FIXED CLIP: " << id << " " << fixed_clip_idx_ << std::endl;
}

void MocapEnv::set_clip_features(TensorMap const &features) {
  reset_next_step_ = true;
  clip_reference_features_ = nullptr;

  fixed_clip_idx_ = -2;
  fixed_clip_features_.clear();
  for (auto const &it : features) {
    auto v = it.second;
    if (it.first == "appendages" || it.first == "body_positions") {
      v = v.view({v.sizes()[0], -1, 3});
    } else if (it.first == "body_quaternions") {
      v = v.view({v.sizes()[0], -1, 4});
    }
    if (verbose_) {
      std::cout << "fixed ref " << it.first << " " << v.sizes() << std::endl;
    }
    fixed_clip_features_[it.first] = v.to(th::kFloat32);
  }
}

void MocapEnv::set_early_termination(bool val) { early_termination_ = val; }

void MocapEnv::set_init_noise(float val) { init_noise_ = val; }

void MocapEnv::set_init_offset(std::pair<int, int> val) { init_offset_ = val; }

void MocapEnv::set_end_with_mocap(bool val) { end_with_mocap_ = val; }

void MocapEnv::set_reward_type(std::string const &val) {
  if (val != "comic" && val != "alive") {
    throw std::runtime_error("Unknown reward type: " + val);
  }
  reward_type_ = val;
}

void MocapEnv::set_init_state(std::string const &val) {
  if (val != "mocap" && val != "default" && val != "hybrid") {
    throw std::runtime_error("Unknown init state: " + val);
  }
  init_state_ = val;
}

std::tuple<TensorMap, bool, double, bool, RewardMap> MocapEnv::reset() {
  // Hooks
  get_clip_to_track();

  // Reset simulation
  mj_resetData(m_, d_);
  auto bak = m_->opt.disableflags;
  m_->opt.disableflags |= mjDSBL_ACTUATION;
  mj_forward(m_, d_);
  m_->opt.disableflags = bak;

  first_update_features_ = true; // just to be sure...
  reset_next_step_ = false;
  tick_ = 0;

  // From Unicon: reset character slightly behind reference frame
  auto reset_step = time_step_;
  if (init_offset_.first != 0 || init_offset_.second != 0) {
    auto offset =
        th::randint(init_offset_.first, init_offset_.second, {1}, rng_)
            .item<int64_t>();
    reset_step = time_step_ + offset;
    if (reset_step < 0) {
      reset_step = 0;
    }
  }

  // set_walker_from_features
  th::Tensor full_qpos, full_qvel;
  auto init_state = init_state_;
  if (init_state == "hybrid") {
    auto choice = th::randint(0, 2, {1}, rng_).item<int64_t>();
    init_state = (choice ? "mocap" : "default");
  }
  if (init_state == "mocap") {
    full_qpos =
        th::hstack({
                       clip_reference_features_->at("position")[reset_step],
                       clip_reference_features_->at("quaternion")[reset_step],
                       clip_reference_features_->at("joints")[reset_step],
                   })
            .to(th::kFloat64);
    full_qvel =
        th::hstack(
            {
                clip_reference_features_->at("velocity")[reset_step],
                clip_reference_features_->at("angular_velocity")[reset_step],
                clip_reference_features_->at("joints_velocity")[reset_step],
            })
            .to(th::kFloat64);
  } else if (init_state == "default") {
    full_qpos = th::zeros({63}, th::kFloat64);
    full_qpos[2] = 1.2;
    full_qpos[3] = 0.859;
    full_qpos[4] = 1.0;
    full_qpos[5] = 1.0;
    full_qpos[6] = 0.859;
    full_qvel = th::zeros({62}, th::kFloat64);
  } else {
    throw std::runtime_error("Invalid init_state");
  }

  if (init_noise_ > 0) {
    // Add noise like in transfer tasks
    auto noise = full_qpos.clone();
    noise.uniform_(-init_noise_, init_noise_, rng_);
    full_qpos.add_(noise);
    noise = th::randn(full_qvel.sizes(), rng_, full_qvel.options());
    full_qvel.add_(init_noise_ * noise);
  }
  auto n_qpos = full_qpos.sizes()[0];
  auto n_qvel = full_qvel.sizes()[0];
  dt_.qpos.narrow(0, 0, n_qpos).copy_(full_qpos);
  dt_.qvel.narrow(0, 0, n_qvel).copy_(full_qvel);
  if (ghost_offset_.defined()) {
    dt_.qpos.narrow(0, n_qpos, n_qpos).copy_(full_qpos);
    dt_.qpos.narrow(0, n_qpos, 3).add_(ghost_offset_);
    dt_.qvel.narrow(0, n_qvel, n_qvel).copy_(full_qvel);
  }
  mj_kinematics(m_, d_);
  bak = m_->opt.disableflags;
  m_->opt.disableflags |= mjDSBL_ACTUATION;
  mj_forward(m_, d_);
  m_->opt.disableflags = bak;

  update_features();
  save_prev_features();
  reference_observations_ = get_all_reference_observations();

  RewardMap rewards = {{"com_reward", 0},
                       {"vel_reward", 0},
                       {"app_reward", 0},
                       {"quat_reward", 0},
                       {"term_reward", 0}};
  return std::make_tuple(get_observation(), true, 0, false, rewards);
}

std::tuple<TensorMap, bool, double, bool, RewardMap>
MocapEnv::step(th::Tensor actions) {
  if (reset_next_step_) {
    //    std::cout << "reset_next_step_!" << std::endl;
    reset_next_step_ = false;
    return reset();
  }

  // self._hooks.before_step
  dt_.ctrl.copy_(actions.to(th::kFloat64));
  // self._observation_updater.prepare_for_next_control_step()

  save_prev_features();

  mjWarningStat prev_warnings[mjNWARNING];
  std::string warnings_str;
  auto physics_is_divergent = false;
  for (auto i = 0; i < n_sub_steps_; i++) {
    memcpy(prev_warnings, d_->warning, sizeof(prev_warnings));
    if (m_->opt.integrator == mjINT_EULER) {
      mj_step2(m_, d_);
    } else {
      mj_step(m_, d_);
    }
    mj_step1(m_, d_);
    // Check warnings
    for (auto j = 0; j < mjNWARNING; j++) {
      if (d_->warning[j].number > prev_warnings[j].number) {
        if (warnings_str.size() > 0) {
          warnings_str += ",";
        }
        warnings_str += g_warning_names[j];
      }
    }
    if (!warnings_str.empty()) {
      if (raise_exception_on_physics_error_) {
        throw std::runtime_error(
            "Physics state is invalid. Warning(s) raised: " + warnings_str);
      } else {
        std::cerr << "Physics state is invalid. Warning(s) raised: " +
                         warnings_str
                  << std::endl;
        physics_is_divergent = true;
        break;
      }
    }
    mj_subtreeVel(m_, d_);
  }

  // One last forward to receive up-to-data observations
  mj_forward(m_, d_);

  // self._hooks.after_step
  if (!end_mocap_) {
    time_step_++;
  }
  tick_++;
  update_features();
  //  np.array(physics.bind(self._walker.mocap_joints).qpos)
  current_reference_features_.clear();
  for (auto const &[k, v] : *clip_reference_features_) {
    current_reference_features_[k] = v[time_step_];
  }
  compute_termination_error(m_, &dt_);
  end_mocap_ = time_step_ == last_step_;
  reference_observations_ = get_all_reference_observations();

  RewardMap rewards;
  auto reward = 0.0f;
  auto terminating = true;
  if (!physics_is_divergent) {
    rewards = get_reward(m_, d_);
    reward = rewards["reward"];
    rewards.erase("reward");
    terminating = (d_->time > time_limit_) || should_terminate_episode();
  }

  if (terminating) {
    reset_next_step_ = true;
  }

  // Update ghost
  if (ghost_offset_.defined()) {
    auto full_qpos = th::hstack({
        clip_reference_features_->at("position")[time_step_],
        clip_reference_features_->at("quaternion")[time_step_],
        clip_reference_features_->at("joints")[time_step_],
    });
    auto full_qvel = th::hstack({
        clip_reference_features_->at("velocity")[time_step_],
        clip_reference_features_->at("angular_velocity")[time_step_],
        clip_reference_features_->at("joints_velocity")[time_step_],
    });
    auto n_qpos = full_qpos.sizes()[0];
    auto n_qvel = full_qvel.sizes()[0];
    dt_.qpos.narrow(0, n_qpos, n_qpos).copy_(full_qpos);
    dt_.qpos.narrow(0, n_qpos, 3).add_(ghost_offset_);
    dt_.qvel.narrow(0, n_qvel, n_qvel).copy_(full_qvel);
  }

  auto obs = get_observation();
  return std::make_tuple(std::move(obs), false, reward, terminating,
                         std::move(rewards));
}

th::Tensor MocapEnv::render(int width, int height) {
  if (rctx_ == nullptr) {
    rctx_ = std::make_unique<MjRenderContext>(m_, 1);
  }
  return rctx_->render(m_, d_, width, height);
}

void MocapEnv::recompile_physics_and_update_observables() {
  if (m_ == nullptr) {
    // mjcf.Physics.from_mjcf_model()
    activate();

    // _temporary_vfs(filenames_and_contents)
    auto fn = "model.xml";
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    auto ret = mj_makeEmptyFileVFS(&vfs, fn, model_xml_.size() + 1);
    if (ret != 0) {
      throw std::runtime_error("Cannot create temporary file");
    }
    auto idx = mj_findFileVFS(&vfs, fn);
    if (idx < 0) {
      throw std::runtime_error("Cannot find temporary file");
    }
    auto vf = vfs.filedata[idx];
    memcpy(vf, model_xml_.c_str(), model_xml_.size());
    static_cast<char *>(vf)[model_xml_.size()] = 0;

    char errbuf[1000];
    m_ = mj_loadXML(fn, &vfs, errbuf, sizeof(errbuf));
    if (m_ == nullptr) {
      throw std::runtime_error(errbuf);
    }

    d_ = mj_makeData(m_);
    if (d_ == nullptr) {
      mj_deleteModel(m_);
      m_ = nullptr;
      throw std::runtime_error("Cannot allocate data");
    }

    dt_ = MjDataTensors(m_, d_);

    // Update kinematics
    mj_forward(m_, d_);

    mj_deleteVFS(&vfs);
  } else {
    // nothing to do?
  }
}

void MocapEnv::after_compile() {
  // Task hook
  if (reference_observations_.size() == 0) {
    reference_observations_ = get_all_reference_observations();
  }
}

void MocapEnv::update_features() {
  dt_.updateFloat32();

  if (first_update_features_) {
    walker_joints_ = dt_.qpos32.narrow(0, 7, m_->nu);
    walker_features_["position"] = dt_.qpos32.narrow(0, 0, 3);
    walker_features_["quaternion"] = dt_.qpos32.narrow(0, 3, 4);
    walker_features_["joints"] = dt_.qpos32.narrow(0, 7, m_->nu);
    walker_features_["center_of_mass"] = dt_.subtree_com32[1];
    walker_features_["velocity"] = dt_.qvel32.narrow(0, 0, 3);
    walker_features_["angular_velocity"] = dt_.qvel32.narrow(0, 3, 3);
    walker_features_["joints_velocity"] = dt_.qvel32.narrow(0, 6, m_->nu);
    walker_features_["body_positions"] = dt_.xpos32.index({xpos_quat_idxs_});
    walker_features_["body_quaternions"] = dt_.xquat32.index({xpos_quat_idxs_});

    first_update_features_ = false;
  } else {
    walker_features_["body_positions"] = dt_.xpos32.index({xpos_quat_idxs_});
    walker_features_["body_quaternions"] = dt_.xquat32.index({xpos_quat_idxs_});
  }

  auto ret = TensorMap();
  // TODO -- umm, just consolidate these two?
  { // end_effectors_pos
    auto end_effectors = dt_.xpos32.index({end_effectors_idx_});
    auto torso = dt_.xpos32[2];
    auto xmat = dt_.xmat32[2].view({3, 3});
    walker_features_["end_effectors"] = (end_effectors - torso).matmul(xmat);
  }
  { // appendages_pos
    // end_effectors_pos with the head
    auto end_effectors = dt_.xpos32.index({appendages_idx_});
    auto torso = dt_.xpos32[2];
    auto xmat = dt_.xmat32[2].view({3, 3});
    walker_features_["appendages"] = (end_effectors - torso).matmul(xmat);
  }
}

void MocapEnv::save_prev_features() {
  for (auto const &k : {"quaternion", "position", "joints"}) {
    walker_features_prev_[k] = walker_features_[k].clone();
  }
}

void MocapEnv::compute_termination_error(mjModel *m, MjDataTensors *dt) {
  auto target_joints = clip_reference_features_->at("joints")[time_step_];
  auto error_joints = th::abs(target_joints - walker_joints_)
                          .masked_select(joints_ref_mask_)
                          .mean()
                          .item<float>();
  auto target_bodies =
      clip_reference_features_->at("body_positions")[time_step_];
  auto error_bodies =
      th::abs(target_bodies - walker_features_["body_positions"])
          .mean()
          .item<float>();
  termination_error_ = (0.5 * error_bodies + 0.5 * error_joints);
}

TensorMap MocapEnv::get_all_reference_observations() {
  auto ret = TensorMap();
  time_steps_.copy_(tref_steps_);
  time_steps_.add_(time_step_ - (time_step_ % ref_lag_));
  time_steps_8x_.copy_(tref_steps_8x_);
  time_steps_8x_.add_(time_step_ - (time_step_ % ref_lag_));
  if (time_steps_[0].item<int64_t>() < 0) {
    // This occurs if past reference frames are included -- simply clamp to
    // the first frame.
    time_steps_.clamp_(0);
    time_steps_8x_.clamp_(0);
  }
  if (clip_reference_features_->at("joints").sizes()[0] <=
      time_step_ + max_ref_step_) {
    throw std::runtime_error("Clip " + clipd_->clip_ids[clip_idx_] +
                             " has insufficient number of features");
  }

  if (observables_set_.count("reference_lag_offset") > 0) {
    ret["reference_lag_offset"] =
        th::tensor({float(time_step_ % ref_lag_) / ref_lag_});
  }
  if (observables_set_.count("reference_rel_bodies_pos_local") > 0) {
    ret["reference_rel_bodies_pos_local"] =
        get_reference_rel_bodies_pos_local();
  }
  if (observables_set_.count("reference_rel_joints") > 0) {
    ret["reference_rel_joints"] = get_reference_rel_joints();
  }
  if (observables_set_.count("reference_rel_bodies_pos_global") > 0) {
    ret["reference_rel_bodies_pos_global"] =
        get_reference_rel_bodies_pos_global();
  }
  if (observables_set_.count("reference_ego_bodies_quats") > 0) {
    ret["reference_ego_bodies_quats"] = get_reference_ego_bodies_quats();
  }
  if (observables_set_.count("reference_rel_root_quat") > 0) {
    ret["reference_rel_root_quat"] = get_reference_rel_root_quat();
  }
  if (observables_set_.count("reference_rel_bodies_quats") > 0) {
    ret["reference_rel_bodies_quats"] = get_reference_rel_bodies_quats();
  }
  if (observables_set_.count("reference_rel_root_pos_local") > 0) {
    ret["reference_rel_root_pos_local"] = get_reference_rel_root_pos_local();
  }
  if (observables_set_.count("ref_relxypos_rel_8x") > 0) {
    // ref_relxypos_8x, with first frame including the delta of the current
    // root position to the reference
    auto obs = clip_reference_features_->at("ref_relxypos_8x")
                   .index({time_steps_8x_})
                   .clone();
    auto xmat = dt_.xmat32[2].view({3, 3});
    auto target_xy = clip_reference_features_->at("ref_rpos_8x")
                         .index({time_steps_8x_})
                         .view({tref_steps_8x_.sizes()[0], -1})
                         .index({0, Slice(0, 2)});
    auto delta_xy = target_xy - walker_features_["position"].narrow(0, 0, 2);
    obs[0].narrow(0, 0, 2).copy_(delta_xy);
    if (reference_dims_ == 2) {
      obs = obs.view({tref_steps_8x_.sizes()[0], -1});
    }
    ret["ref_relxypos_rel_8x"] = obs;
  }
  if (observables_set_.count("ref_relxypos_rel") > 0) {
    // ref_relxypos_8x, with first frame including the delta of the current
    // root position to the reference
    auto obs = clip_reference_features_->at("ref_relxypos")
                   .index({time_steps_})
                   .clone();
    auto xmat = dt_.xmat32[2].view({3, 3});
    auto target_xy = clip_reference_features_->at("ref_rpos")
                         .index({time_steps_})
                         .view({tref_steps_.sizes()[0], -1})
                         .index({0, Slice(0, 2)});
    auto delta_xy = target_xy - walker_features_["position"].narrow(0, 0, 2);
    obs[0].narrow(0, 0, 2).copy_(delta_xy);
    if (reference_dims_ == 2) {
      obs = obs.view({tref_steps_.sizes()[0], -1});
    }
    ret["ref_relxypos_rel"] = obs;
  }
  return ret;
}

th::Tensor MocapEnv::get_reference_rel_bodies_pos_local() {
  auto xmat = dt_.xmat32[2].view({3, 3});
  auto obs =
      (clip_reference_features_->at("body_positions").index({time_steps_}) -
       walker_features_["body_positions"])
          .index({Slice(), body_idxs_})
          .matmul(xmat);
  return obs;
}

th::Tensor MocapEnv::get_reference_rel_joints() {
  auto diff = (clip_reference_features_->at("joints").index({time_steps_}) -
               walker_joints_);
  return diff.index({Slice(), mocap_to_observable_joint_order_});
}

th::Tensor MocapEnv::get_reference_rel_bodies_pos_global() {
  return (clip_reference_features_->at("body_positions").index({time_steps_}) -
          walker_features_["body_positions"])
      .index({Slice(), body_idxs_});
}

th::Tensor MocapEnv::get_reference_ego_bodies_quats() {
  std::vector<th::Tensor> obs;
  auto quats_for_clip = reference_ego_bodies_quats_[clip_idx_];
  for (auto t : ref_steps_) {
    auto tt = t + time_step_;
    if (quats_for_clip.find(tt) == quats_for_clip.end()) {
      auto root_quat = clip_reference_features_->at("quaternion")[tt];
      quats_for_clip[tt] =
          quat_diff(root_quat, clip_reference_features_->at("body_quaternions")
                                   .index({tt, body_idxs_}));
    }
    obs.push_back(quats_for_clip[tt].view({-1}));
  }
  return th::stack(obs);
}

th::Tensor MocapEnv::get_reference_rel_root_quat() {
  return quat_diff(
      walker_features_["quaternion"],
      clip_reference_features_->at("quaternion").index({time_steps_}));
}

th::Tensor MocapEnv::get_reference_rel_bodies_quats() {
  return quat_diff(walker_features_["body_quaternions"].index({body_idxs_}),
                   clip_reference_features_->at("body_quaternions")
                       .index({time_steps_})
                       .index({Slice(), body_idxs_}));
}

th::Tensor MocapEnv::get_reference_rel_root_pos_local() {
  auto xmat = dt_.xmat32[2].view({3, 3});
  auto obs = (clip_reference_features_->at("position").index({time_steps_}) -
              walker_features_["position"])
                 .matmul(xmat);
  return obs;
}

th::Tensor MocapEnv::get_single_observation(Observation o) {
  switch (o) {
  case Observation::ActuatorActivation:
    return dt_.act32;
  case Observation::AppendagesPos: {
    return walker_features_["appendages"].view({-1});
  }
  case Observation::BodyHeight: {
    auto acc = dt_.xpos32.accessor<float, 2>();
    return th::tensor({acc[2][2]});
  }
  case Observation::BodyQuaternions:
    return walker_features_["body_quaternions"].view({-1});
  case Observation::ClipID:
    return th::tensor({clip_idx_}, th::kInt64);
  case Observation::EndEffectorsPos:
    return walker_features_["end_effectors"].view({-1});
  case Observation::Frame:
    return th::tensor({time_step_}, th::kInt64);
  case Observation::GyroControl: {
    auto normed_diff = quat_diff(walker_features_prev_["quaternion"],
                                 walker_features_["quaternion"]);
    normed_diff.div_(normed_diff.norm());
    auto obs = quat_to_axisangle(normed_diff);
    obs.div_(control_timestep_);
    return obs;
  }
  case Observation::JointsPos:
    // The dm_control walker observable will return this with a differen
    // ordering
    return walker_features_["joints"];
  case Observation::JointsVel:
    // The dm_control walker observable will return this with a differen
    // ordering
    return walker_features_["joints_velocity"];
  case Observation::JointsVelControl: {
    auto obs = (walker_features_["joints"] - walker_features_prev_["joints"])
                   .index({mocap_to_observable_joint_order_});
    obs.div_(control_timestep_);
    return obs;
  }
  case Observation::QPos:
    return dt_.qpos32;
  case Observation::QVel:
    return dt_.qvel32;
  case Observation::Quaternion:
    return walker_features_["quaternion"];
  case Observation::ReferenceAppendagesPos: {
    auto obs = clip_reference_features_->at("appendages").index({time_steps_});
    if (reference_dims_ == 1) {
      return obs.view({-1});
    }
    return obs.view({obs.sizes()[0], -1});
  }
  case Observation::ReferenceRelRootPosLocalCurrent: {
    auto xmat = dt_.xmat32[2].view({3, 3});
    auto obs = (clip_reference_features_->at("position")[time_step_] -
                walker_features_["position"])
                   .matmul(xmat);
    return obs;
  }
  case Observation::ReferenceRelRootXYPosCurrent:
    return (clip_reference_features_->at("position")[time_step_] -
            walker_features_["position"])
        .narrow(0, 0, 2);
  case Observation::ReferenceRelRootQuatCurrent:
    return quat_diff(walker_features_["quaternion"],
                     clip_reference_features_->at("quaternion")[time_step_]);
  case Observation::RefJointsPos:
    return clip_reference_features_->at("joints")[time_step_];
  case Observation::RefJointsVel:
    return clip_reference_features_->at("joints_velocity")[time_step_];
  case Observation::RefAppendages:
    return clip_reference_features_->at("appendages")[time_step_].view({-1});
  case Observation::RefVelocity:
    return clip_reference_features_->at("velocity")[time_step_];
  case Observation::RefAngularVelocity:
    return clip_reference_features_->at("angular_velocity")[time_step_];
  case Observation::RootAngularVelocity:
    return walker_features_["angular_velocity"];
  case Observation::RootVelocity:
    return walker_features_["velocity"];
  case Observation::SensorsAccelerometer:
    return dt_.sensordata32.narrow(0, 6, 3);
  case Observation::SensorsGyro:
    return dt_.sensordata32.narrow(0, 3, 3);
  case Observation::SensorsForce:
    // empty
    return {};
  case Observation::SensorsTorque: {
    auto const _TORQUE_THRESHOLD = 60;
    return th::tanh(2 * dt_.sensordata32.narrow(0, 19, 6) / _TORQUE_THRESHOLD);
  }
  case Observation::SensorsTouch: {
    auto const _TOUCH_THRESHOLD = 1e-3;
    return (dt_.sensordata32.narrow(0, 9, 10) > _TOUCH_THRESHOLD)
        .to(th::kFloat32);
  }
  case Observation::SensorsVelocimeter:
    return dt_.sensordata32.narrow(0, 0, 3);
  case Observation::Tick:
    return th::tensor({tick_}, th::kInt64);
  case Observation::TimeInClip: {
    auto normalized_time =
        (current_start_time_ + d_->time) / clipd_->clips[clip_idx_].duration;
    return th::tensor({normalized_time}, th::kFloat32);
  }
  case Observation::WorldZAxis:
    return dt_.xmat32.index({2, Slice(6, 9)});
  case Observation::VelocimeterControl: {
    auto rmat_prev = quat_to_mat32(walker_features_prev_["quaternion"]);
    auto veloc_world =
        (walker_features_["position"] - walker_features_prev_["position"]);
    veloc_world.div_(control_timestep_);
    return veloc_world.matmul(rmat_prev);
  }
  default:
    break;
  }

  if (custom_observables_.count(o) == 0) {
    throw std::runtime_error("Unknown feature: " + std::to_string(o));
  }
  auto o_str = custom_observables_[o];

  if (reference_observations_.count(o_str) > 0) {
    auto obs = reference_observations_[o_str];
    if (reference_dims_ == 1) {
      return obs.view({-1});
    }
    return obs.view({obs.sizes()[0], -1});
  } else if (clip_reference_features_->count(o_str) > 0) {
    th::Tensor obs;
    try {
      if (ends_with(o_str, "_8x")) {
        obs = clip_reference_features_->at(o_str).index({time_steps_8x_});
        if (reference_dims_ == 2) {
          obs = obs.view({tref_steps_8x_.sizes()[0], -1});
        } else {
          obs = obs.view({-1});
        }
      } else {
        obs = clip_reference_features_->at(o_str).index({time_steps_});
        if (reference_dims_ == 2) {
          obs = obs.view({tref_steps_.sizes()[0], -1});
        } else {
          obs = obs.view({-1});
        }
      }
    } catch (...) {
      std::cerr << "Exception obtaining clip reference for clip "
                << clipd_->clip_ids[clip_idx_] << std::endl;
      throw;
    }
    return obs;
  }
  throw std::runtime_error("Unsupported observation " + o_str);
}

TensorMap MocapEnv::get_observation() {
  auto obs = TensorMap();
  for (auto const &[k, v] : observables_) {
    if (v.size() == 1) {
      obs[k] = get_single_observation(v[0]);
    } else {
      std::vector<th::Tensor> tobs;
      for (auto const &key : v) {
        tobs.emplace_back(get_single_observation(key));
      }
      obs[k] = th::cat(tobs, -1);
    }
  }
  return obs;
}

void MocapEnv::get_clip_to_track() {
  int idx;
  if (fixed_clip_idx_ == -2) {
    // TODO sample starting positions with fixed clip features?
    clip_reference_features_ = &fixed_clip_features_;
    time_step_ = 0;
    current_start_time_ = 0;
    last_step_ =
        clip_reference_features_->at("joints").sizes()[0] - max_ref_step_ - 1;
    return;
  }

  while (true) {
    idx = th::multinomial(clipd_->start_probabilities, 1, false, rng_)
              .item<int64_t>();
    clip_idx_ = clipd_->possible_starts[idx].first;
    if (fixed_clip_idx_ >= 0 && clip_idx_ != fixed_clip_idx_) {
      continue;
    }
    break;
  }
  auto start_step = clipd_->possible_starts[idx].second;
  clip_reference_features_ = &clipd_->clips[clip_idx_].data32;

  // _strip_reference_prefix() is a no-op for this dataset
  // TODO verify true for all dataset?
  time_step_ = start_step - clipd_->clips[clip_idx_].start_steps;
  current_start_time_ = time_step_ * clipd_->clips[clip_idx_].dt;
  last_step_ = clipd_->clips[clip_idx_].num_frames() - max_ref_step_ - 1;
  if (verbose_) {
    std::cout << "Mocap " << clipd_->clip_ids[clip_idx_] << " at step "
              << start_step << " with remaining length "
              << last_step_ - start_step << "; time_step_ = " << time_step_
              << std::endl;
  }
}

RewardMap MocapEnv::get_reward(mjModel *m, mjData *d) {
  if (reward_type_ == "alive") {
    return {{"reward", 1.0f}};
  }

  // comic_reward_fn()
  auto termination_reward =
      1.0 - termination_error_ / termination_error_threshold_;
  // multi_term_pose_reward_fn
  std::unordered_map<std::string, float> sqdiffs;
  auto &ref_features = current_reference_features_;

  sqdiffs["center_of_mass"] =
      (walker_features_["center_of_mass"] - ref_features["center_of_mass"])
          .square()
          .sum()
          .item<float>();
  sqdiffs["joints_velocity"] =
      (joints_ref_mask_ *
       (walker_features_["joints_velocity"] - ref_features["joints_velocity"]))
          .square()
          .sum()
          .item<float>();
  sqdiffs["appendages"] =
      (walker_features_["appendages"] - ref_features["appendages"])
          .square()
          .sum()
          .item<float>();
  sqdiffs["body_quaternions"] =
      bounded_quat_dist2(walker_features_["body_quaternions"],
                         ref_features["body_quaternions"])
          .square()
          .sum()
          .item<float>();

  auto com = 0.1f * std::exp(-10.0f * sqdiffs["center_of_mass"]);
  auto joints_velocity = 1.0f * std::exp(-0.1f * sqdiffs["joints_velocity"]);
  auto appendages = 0.15f * std::exp(-40.0f * sqdiffs["appendages"]);
  auto body_quaternions = 0.65f * std::exp(-2.0f * sqdiffs["body_quaternions"]);
  auto mt_reward = com + joints_velocity + appendages + body_quaternions;
  auto reward = 0.5f * termination_reward + 0.5f * mt_reward;
  return {{"com_reward", 0.5f * com},
          {"vel_reward", 0.5f * joints_velocity},
          {"app_reward", 0.5f * appendages},
          {"quat_reward", 0.5f * body_quaternions},
          {"term_reward", 0.5f * termination_reward},
          {"reward", reward}};
}

bool MocapEnv::should_terminate_episode() const {
  if (early_termination_ && termination_error_ > termination_error_threshold_) {
    return true;
  }
  if (end_with_mocap_) {
    return end_mocap_;
  }

  // End on contact of any body part (apart from feet) touching the floor
  for (auto i = 0; i < d_->ncon; i++) {
    if (d_->contact[i].geom1 != 0) {
      continue;
    }
    auto g2 = d_->contact[i].geom2;
    if (g2 == 8 || g2 == 9 || g2 == 10) { // ltoes
      continue;
    }
    if (g2 == 17 || g2 == 18 || g2 == 19) { // rtoes
      continue;
    }
    if (g2 == 6 || g2 == 7) { // lfoot
      continue;
    }
    if (g2 == 15 || g2 == 16) { // rfoot
      continue;
    }
    return true;
  }

  return false;
}

// Some more TODOs:
// - Try manual threads with busy waiting instead of the thread pool -- maybe
// - latency still kinda sucks?
// - Pull together step() and next_step()
// - Return dictionary again
// - Finally, verify!!!
BatchedMocapEnv::BatchedMocapEnv(std::shared_ptr<ClipData> clipd,
                                 TaskData const &taskd, float time_limit,
                                 int num_envs, int device, int reference_dims,
                                 int ref_lag, bool verbose)
    : clipd_(clipd), observables_(taskd.observables), th_barrier_(num_envs + 1),
      verbose_(verbose) {
  if (num_envs < 1) {
    throw std::runtime_error("Expect num_envs >= 1");
  }

  // Ask environments to return each observation by its own to optimize copies
  // into static buffers.
  std::unordered_map<std::string, std::vector<std::string>> env_observables;
  for (auto const &it : observables_) {
    for (auto const &obs : it.second) {
      env_observables[obs] = {obs};
    }
  }

  th_action_.resize(num_envs);
  if (verbose_) {
    std::cout << "Constructing " << num_envs << " MocapEnv instances"
              << std::endl;
  }
  for (auto i = 0; i < num_envs; i++) {
    envs_.emplace_back(std::make_unique<MocapEnv>(clipd, taskd, time_limit,
                                                  reference_dims, ref_lag,
                                                  env_observables, verbose_));
    threads_.emplace_back(&BatchedMocapEnv::run, this, i);
  }
  if (verbose_) {
    std::cout << "Done" << std::endl;
  }
  done_ = th::zeros({num_envs}, th::kBool);
  rewards_ = th::empty({num_envs}, th::kFloat32);
  for (auto const &key : {"com", "vel", "app", "quat", "term"}) {
    reward_terms_[std::string(key) + "_reward"] =
        th::empty({num_envs}, th::kFloat32);
  }
#ifdef HAVE_CUDA
  done_ = done_.pin_memory();
  rewards_ = rewards_.pin_memory();
  for (auto &it : reward_terms_) {
    it.second = it.second.pin_memory();
  }
  ret_opts_ = th::TensorOptions().device(torch::kCUDA, device);
#else  // HAVE_CUDA
  ret_opts_ = th::TensorOptions();
#endif // HAVE_CUDA

  if (verbose_) {
    std::cout << "ctor done" << std::endl;
  }
}

BatchedMocapEnv::~BatchedMocapEnv() {
  {
    std::unique_lock<std::mutex> lock(th_mutex_);
    for (auto i = 0UL; i < threads_.size(); i++) {
      th_action_[i] = "stop";
    }
    th_cvar_.notify_all();
  }
  for (auto &t : threads_) {
    t.join();
  }
}

TensorMap BatchedMocapEnv::observation_spec() {
  if (flattened_obs_.empty()) {
    // XXX
    reset();
  }
  TensorMap ret;
  for (auto const &[key, val] : flattened_obs_) {
    // XXX hack
    if (key == "clip_id") {
      ret[key] = th::zeros_like(val[0]) + int(clipd_->clips.size());
      // XXX hack
    } else if (key == "tick" || key == "frame") {
      auto longest =
          std::max_element(clipd_->clips.begin(), clipd_->clips.end(),
                           [](Clip const &c1, Clip const &c2) {
                             return c1.num_frames() < c2.num_frames();
                           });
      ret[key] = th::zeros_like(val[0]) + int(longest->num_frames());
    } else {
      ret[key] = th::zeros_like(val[0]);
    }
  }

  return ret;
}

TensorMap BatchedMocapEnv::stack_obs(std::vector<TensorMap> const &obs) {
  auto ret = TensorMap();
  if (obs.empty()) {
    return ret;
  }
  for (auto const &it : obs[0]) {
    std::vector<th::Tensor> t;
    for (auto const &map : obs) {
      t.push_back(map.at(it.first));
    }
    ret[it.first] = th::stack(t);
  }
  return ret;
}

TensorMap BatchedMocapEnv::reset() {
  if (last_obs_.empty()) {
    // XXX
    step(th::zeros({int64_t(envs_.size()), 56}));
  }

  {
    std::unique_lock<std::mutex> lock(th_mutex_);
    for (auto i = 0UL; i < threads_.size(); i++) {
      th_action_[i] = "reset";
    }
    th_cvar_.notify_all();
  }
  th_barrier_.wait();

  done_.zero_();

  TensorMap obs_ret;
  for (auto const &[k, v] : flattened_obs_) {
    obs_ret[k] = v.to(ret_opts_.dtype(v.dtype()), true);
  }
  return obs_ret;
}

TensorMap BatchedMocapEnv::reset_if_done() {
  if (last_obs_.empty()) {
    throw std::runtime_error(
        "step() needs to be called before reset_if_done()");
  }

  {
    std::unique_lock<std::mutex> lock(th_mutex_);
    for (auto i = 0UL; i < threads_.size(); i++) {
      th_action_[i] = "reset_if_done";
    }
    th_cvar_.notify_all();
  }
  th_barrier_.wait();

  TensorMap obs_ret;
  for (auto const &[k, v] : flattened_obs_) {
    obs_ret[k] = v.to(ret_opts_, true);
  }
  return obs_ret;
}

std::tuple<TensorMap, th::Tensor, th::Tensor, TensorMap>
BatchedMocapEnv::step(th::Tensor actions) {
  {
    std::unique_lock<std::mutex> lock(th_mutex_);
#ifdef HAVE_CUDA
    if (actions_.numel() == 0) {
      actions_ = actions.to(th::kCPU).pin_memory();
    } else {
      actions_.copy_(actions.to(th::kCPU), true);
    }
#else
    actions_ = actions.to(th::kCPU, th::kFloat64);
#endif
    for (auto i = 0UL; i < threads_.size(); i++) {
      th_action_[i] = "step";
    }
    th_cvar_.notify_all();
  }
  th_barrier_.wait();

  if (last_obs_.empty()) {
    std::vector<TensorMap> iobs;
    for (auto &it : initial_obs_) {
      iobs.emplace_back(std::move(it.second));
    }
    auto stacked = stack_obs(iobs);

    // Allocate flattened observation(s)
    for (auto const &it : observables_) {
      auto const &key = it.first;
      int64_t n = 0;
      int64_t c = 0;
      bool has_2d = false;
      auto dtype = th::kFloat32;
      for (auto const &obs : it.second) {
        int64_t numel = 1;
        for (auto i = 1U; i < stacked[obs].dim(); i++) {
          numel *= stacked[obs].sizes()[i];
        }
        if (verbose_) {
          std::cout << key << "/" << obs << " " << stacked[obs].sizes()
                    << std::endl;
        }
        if (n == 0) {
          has_2d = (stacked[obs].dim() > 2);
        } else if (has_2d != (stacked[obs].dim() > 2)) {
          throw std::runtime_error("Iconsistent dimensions for observation: " +
                                   key + "; has_2d=" + std::to_string(has_2d));
        }
        n += numel;
        c += stacked[obs].sizes().back();
        dtype = stacked[obs].scalar_type();
      }
      auto opts = th::TensorOptions().dtype(dtype);
      if (has_2d) {
        if (n % c != 0) {
          throw std::runtime_error("Iconsistent shape for observation: " + key);
        }
        flattened_obs_[key] = th::ones({int64_t(envs_.size()), n / c, c}, opts);
      } else {
        flattened_obs_[key] = th::ones({int64_t(envs_.size()), n}, opts);
      }
      if (dtype != th::kInt64) {
        flattened_obs_[key].mul_(NAN);
      }
      // Flattened obs is filled with NAN to expose potential issues in the
      // logic below.
#ifdef HAVE_CUDA
      flattened_obs_[key] = flattened_obs_[key].pin_memory();
#endif
      int64_t offset = 0;
      for (auto const &obs : it.second) {
        std::vector<int64_t> shape;
        shape.push_back(envs_.size());
        std::vector<int64_t> strides;
        strides.push_back(n);
        for (auto i = 1U; i < stacked[obs].dim(); i++) {
          shape.push_back(stacked[obs].sizes()[i]);
          strides.push_back(stacked[obs].strides()[i]);
        }
        if (strides.size() > 2) {
          strides[1] = c;
        }
        if (verbose_) {
          std::cout << "obs " << obs << " buffer " << obs << " offset "
                    << offset << " shape " << shape << " strides " << strides
                    << " dtype " << dtype << std::endl;
        }
        auto tptr = th::from_blob(
            static_cast<int8_t *>(flattened_obs_[key].data_ptr()) +
                offset * elementSize(dtype),
            shape, strides, opts);
        last_obs_.insert({obs, tptr});
        offset += shape.back();
      }
      for (auto const &[k, v] : stacked) {
        for (auto i = 0UL; i < iobs.size(); i++) {
          v[i].copy_(iobs[i][k]);
        }
      }
    }
  }

  TensorMap obs_ret;
  for (auto const &[k, v] : flattened_obs_) {
    obs_ret[k] = v.to(ret_opts_.dtype(v.dtype()), true);
  }
  TensorMap rt_ret;
  for (auto const &[k, v] : reward_terms_) {
    rt_ret[k] = v.to(ret_opts_, true);
  }
  auto ret = std::make_tuple(obs_ret, rewards_.to(ret_opts_, true),
                             done_.to(ret_opts_, true), rt_ret);
  return ret;
}

th::Tensor BatchedMocapEnv::render_single(int width, int height, int idx) {
  std::unique_lock<std::mutex> lock(th_mutex_);
  return envs_[idx]->render(width, height);
}

void BatchedMocapEnv::run(int idx) {
  while (true) {
    std::string th_action;
    {
      std::unique_lock<std::mutex> lock(th_mutex_);
      th_cvar_.wait(lock, [this, idx] { return !th_action_[idx].empty(); });
      th_action = th_action_[idx];
      th_action_[idx] = "";
    }
    if (th_action == "stop") {
      break;
    } else if (th_action == "step") {
      auto rewards_acc = rewards_.accessor<float, 1>();
      auto done_acc = done_.accessor<bool, 1>();
      auto [obs, reset, reward, done, reward_terms] =
          envs_[idx]->step(actions_[idx]);
      rewards_acc[idx] = reward;
      done_acc[idx] = done;
      for (auto const &it : reward_terms) {
        reward_terms_[it.first].accessor<float, 1>()[idx] = it.second;
      }
      if (!last_obs_.empty()) {
        for (auto const &[k, v] : obs) {
          auto range = last_obs_.equal_range(k);
          for (auto jt = range.first; jt != range.second; ++jt) {
            jt->second[idx].copy_(v);
          }
        }
      } else {
        std::unique_lock<std::mutex> l2(th_mutex_);
        initial_obs_[idx] = obs;
      }
    } else if (th_action == "reset") {
      auto [obs, reset, reward, done, reward_terms] = envs_[idx]->reset();
      for (auto const &[k, v] : obs) {
        auto range = last_obs_.equal_range(k);
        for (auto jt = range.first; jt != range.second; ++jt) {
          jt->second[idx].copy_(v);
        }
      }
    } else if (th_action == "reset_if_done") {
      auto done_acc = done_.accessor<bool, 1>();
      if (done_acc[idx]) {
        auto [obs, reset, reward, done, reward_terms] = envs_[idx]->reset();
        for (auto const &[k, v] : obs) {
          auto range = last_obs_.equal_range(k);
          for (auto jt = range.first; jt != range.second; ++jt) {
            jt->second[idx].copy_(v);
          }
        }
        done_acc[idx] = 0;
      }
    }

    th_barrier_.wait();
  }
}
