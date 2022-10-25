/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>

#include "common_py.h"

using namespace pybind11::literals;

th::Tensor th_from_numpy(py::buffer buf) {
  auto types = buf.attr("dtype").attr("name").cast<std::string>();
  auto info = buf.request();
  auto is = info.itemsize;
  std::vector<int64_t> shape;
  for (auto s : info.shape) {
    shape.push_back(int64_t(s));
  }
  std::vector<int64_t> strides;
  for (auto s : info.strides) {
    strides.push_back(int64_t(s) / is);
  }
  if (types == "int64") {
    return th::from_blob(info.ptr, at::ArrayRef<int64_t>(shape),
                         at::ArrayRef<int64_t>(strides), th::dtype(th::kInt64));
  } else if (types == "int32") {
    // XXX Hotfix for transposed data? Not sure what's going on here, somewhere
    // in the pipeline it may be transposed... :/
    return th::from_blob(info.ptr, at::ArrayRef<int64_t>(shape),
                         at::ArrayRef<int64_t>(strides), th::dtype(th::kInt32))
        .to(th::kInt64)
        .t();
  } else if (types == "float32") {
    return th::from_blob(info.ptr, at::ArrayRef<int64_t>(shape),
                         at::ArrayRef<int64_t>(strides),
                         th::dtype(th::kFloat32));
  } else if (types == "float64") {
    return th::from_blob(info.ptr, at::ArrayRef<int64_t>(shape),
                         at::ArrayRef<int64_t>(strides),
                         th::dtype(th::kFloat64));
  }
  throw std::runtime_error("Cannot convert data of type " + types);
}

std::vector<std::string> featuresForObservable(std::string observable) {
  static std::unordered_set<std::string> proprio_obs = {
      "appendages_pos",      "body_height",           "joints_pos",
      "joints_vel",          "gyro_control",          "joints_vel_control",
      "velocimeter_control", "sensors_touch",         "sensors_velocimeter",
      "sensors_gyro",        "sensors_accelerometer", "end_effectors_pos",
      "actuator_activation", "sensors_torque",        "world_zaxis",
      "quaternion",
  };
  static std::unordered_set<std::string> static_obs = {
      "reference_lag_offset",
      "frame",
      "clip_id",
      "time_in_clip",
      "tick",
      "qpos",
      "qvel",
      "act",
  };

  if (proprio_obs.count(observable) > 0 || static_obs.count(observable) > 0) {
    // We'll need those to replay the clip in the first place (and for
    // computing the Comic reward)
    return {"position",       "quaternion",       "joints",
            "velocity",       "angular_velocity", "joints_velocity",
            "center_of_mass", "body_positions",   "body_quaternions",
            "appendages"};
  } else if (observable == "reference_rel_bodies_pos_local") {
    return {"body_positions"};
  } else if (observable == "reference_rel_joints") {
    return {"joints"};
  } else if (observable == "reference_rel_bodies_pos_global") {
    return {"body_positions"};
  } else if (observable == "reference_ego_bodies_quats") {
    return {"quaternion", "body_quaternions"};
  } else if (observable == "reference_rel_root_quat") {
    return {"quaternion"};
  } else if (observable == "reference_rel_bodies_quats") {
    return {"body_quaternions"};
  } else if (observable == "reference_rel_root_pos_local") {
    return {"body_positions"};
  } else if (observable == "reference_rel_root_xypos_current") {
    return {"body_positions"};
  } else if (observable == "reference_rel_root_quat_current") {
    return {"body_positions"};
  } else if (observable == "reference_rel_root_pos_local_current") {
    return {"body_positions"};
  } else if (observable == "reference_appendages_pos") {
    return {"appendages"};
  } else if (observable == "ref_relxypos_rel_8x") {
    return {"ref_relxypos_8x", "ref_rpos_8x"};
  } else if (observable == "ref_relxypos_rel") {
    return {"ref_relxypos", "ref_rpos"};
    //  } else if (!observable.compare(0, 4, "ref_")) {
    //    return {observable.substr(4)};
  }
  // Assume this is a raw feature that we'll pass as-is.
  return {observable};
}

th::Tensor loadFeature(py::object h5file, std::string const &id,
                       std::string const &name) {
  auto data = h5file.attr("get")(id.c_str())
                  .attr("get")("walkers")
                  .attr("get")("walker_0")
                  .attr("get")(name.c_str());
  if (data.is_none()) {
    throw std::runtime_error("Cannot find in data file: " + id +
                             "/walkers/walker_0/" + name);
  }
  // Copy data with [:]
  auto npd = data[py::slice(0, std::numeric_limits<int64_t>::max(), 1)];
  auto val = py::reinterpret_borrow<py::buffer>(npd);
  return th_from_numpy(val).clone().t();
}

void parseClipData(
    ClipData &d, py::object task,
    std::unordered_map<std::string, std::vector<std::string>> const
        &observables,
    bool verbose) {
  if (verbose) {
    std::cout << "parseClipData()" << std::endl;
  }
  auto starts = py::list(task.attr("_possible_starts"));
  d.possible_starts.clear();
  for (auto start : starts) {
    auto s = py::reinterpret_borrow<py::tuple>(start);
    d.possible_starts.emplace_back(py::int_(s[0]), py::int_(s[1]));
  }
  d.start_probabilities = th_from_numpy(py::reinterpret_borrow<py::buffer>(
                                            task.attr("_start_probabilities")))
                              .clone();

  std::unordered_set<std::string> features;
  for (auto const &it : observables) {
    for (auto const &obs : it.second) {
      for (auto const &feat : featuresForObservable(obs)) {
        features.insert(feat);
      }
    }
  }

  // TODO this seems a bit bogus -- in the original task we have both a
  // "dataset" (ClipCollection) and "all_clips" (vector of trajectories).
  // Presumably this can ease referencing the same clip multiple times in the
  // dataset? Reflect this! Or not necessary?
  auto dataset = task.attr("_dataset");
  auto ids = py::tuple(dataset.attr("ids"));
  if (verbose) {
    std::cout << "Loading " << ids.size() << " clips..." << std::endl;
  }
  d.clip_ids.clear();
  d.clips.clear();
  for (size_t i = 0; i < ids.size(); i++) {
    d.clip_ids.push_back(py::str(ids[i]));
    auto pyclip = task.attr("_all_clips")[py::int_(i)];
    Clip clip;
    if (verbose) {
      std::cout << d.clip_ids.back() << " " << i << " " << py::str(pyclip)
                << std::endl;
    }
    clip.dt = py::float_(pyclip.attr("dt"));
    clip.duration = py::float_(pyclip.attr("duration"));
    py::dict data = pyclip.attr("as_dict")();
    for (const auto &feat : features) {
      auto tkey = "walker/" + feat;
      if (data.contains(tkey.c_str())) {
        auto val = py::reinterpret_borrow<py::buffer>(data[tkey.c_str()]);
        auto tval = th_from_numpy(val).clone();
        if (i == 0) {
          if (verbose) {
            std::cout << "add preloaded key " << feat << ": " << tval.sizes()
                      << std::endl;
          }
        }
        clip.data[feat] = tval;
      } else {
        auto tval = loadFeature(task.attr("_loader").attr("_h5_file"),
                                py::str(ids[i]), feat);
        if (i == 0) {
          if (verbose) {
            std::cout << "add key " << feat << ": " << tval.sizes()
                      << std::endl;
          }
        }
        clip.data[feat] = tval;
      }
    }

    for (auto const &[k, v] : clip.data) {
      if (v.scalar_type() == th::kInt64 || v.scalar_type() == th::kInt32) {
        clip.data32[k] = v;
      } else {
        clip.data32[k] = v.to(th::kFloat32);
      }
    }

    d.clips.emplace_back(std::move(clip));
  }

  if (verbose) {
    std::cout << "done" << std::endl;
  }
}

void parseTaskData(
    TaskData &d, py::object task,
    std::unordered_map<std::string, std::vector<std::string>> const
        &observables,
    bool verbose) {
  if (verbose) {
    std::cout << "parseTaskData()" << std::endl;
  }
  // TODO assert correct task type
  //    py::python_builtins::isinstance();
  if (!task.attr("_disable_props").cast<bool>()) {
    throw std::runtime_error("Props enabled but support is missing");
  }

  d.max_ref_step = py::int_(task.attr("_max_ref_step"));
  d.ref_steps = py_vector<int>(task.attr("_ref_steps"));
  d.termination_error_threshold =
      py::float_(task.attr("_termination_error_threshold"));
  d.n_sub_steps = py::int_(task.attr("physics_steps_per_control_step"));
  d.control_timestep = py::float_(task.attr("control_timestep"));

  d.observables = observables;

  if (!task.attr("_ghost_offset").is_none()) {
    auto offset = task.attr("_ghost_offset").cast<std::vector<double>>();
    d.ghost_offset = th::tensor(offset, th::kFloat64);
  }

  auto mjcf_model = task.attr("root_entity").attr("mjcf_model");
  auto xml = mjcf_model.attr("to_xml_string")();
  py::dict assets = mjcf_model.attr("get_assets")();
  if (py::len(assets) > 0) {
    throw std::runtime_error("Asset handling not implemented");
  }
  d.model_xml = xml.cast<std::string>();

  d.body_idxs = th_from_numpy(task.attr("_body_idxs")).to(th::kInt64).clone();
  auto joint_order = py_vector<int>(
      task.attr("_walker").attr("mocap_to_observable_joint_order"));
  d.mocap_to_observable_joint_order =
      th::empty({int64_t(joint_order.size())}, th::kInt64);
  for (auto i = 0UL; i < joint_order.size(); i++) {
    d.mocap_to_observable_joint_order[i] = joint_order[i];
  }
}
