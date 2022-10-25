/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <limits>

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "common_py.h"
#include "mocap_env.h"

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<MocapEnv>(m, "MocapEnv")
      .def(py::init([](py::object task, int time_limit,
                       std::unordered_map<std::string, std::vector<std::string>>
                           observables) {
             auto clipd = std::make_shared<ClipData>();
             TaskData taskd;
             parseClipData(*clipd, task, observables);
             parseTaskData(taskd, task, observables);
             return new MocapEnv(std::move(clipd), taskd, time_limit, 1, 1);
           }),
           py::arg("task"), py::arg("time_limit"), py::arg("observables"))
      .def("action_spec", &MocapEnv::action_spec)
      .def("observation_spec", &MocapEnv::observation_spec)
      .def("reset", &MocapEnv::reset)
      .def("seed", &MocapEnv::seed)
      .def("set_clip", &MocapEnv::set_clip)
      .def("set_clip_features", &MocapEnv::set_clip_features)
      .def("set_early_termination", &MocapEnv::set_early_termination)
      .def("set_init_noise", &MocapEnv::set_init_noise)
      .def("set_init_offset", &MocapEnv::set_init_offset)
      .def("set_init_state", &MocapEnv::set_init_state)
      .def("set_end_with_mocap", &MocapEnv::set_end_with_mocap)
      .def("set_reward_type", &MocapEnv::set_reward_type)
      .def("step", &MocapEnv::step)
      //      .def("data", &MocapEnv::data)
      //      .def("model", &MocapEnv::model)
      .def("render", &MocapEnv::render);
  py::class_<BatchedMocapEnv>(m, "BatchedMocapEnv")
      .def(
          py::init([](py::object task, int time_limit, int num_envs, int device,
                      std::unordered_map<std::string, std::vector<std::string>>
                          observables,
                      int reference_dims, int reference_lag, bool verbose) {
            auto clipd = std::make_shared<ClipData>();
            TaskData taskd;
            parseClipData(*clipd, task, observables, verbose);
            parseTaskData(taskd, task, observables, verbose);
            return new BatchedMocapEnv(std::move(clipd), taskd, time_limit,
                                       num_envs, device, reference_dims,
                                       reference_lag, verbose);
          }),
          py::arg("task"), py::arg("time_limit"), py::arg("num_envs"),
          py::arg("device"), py::arg("observables"), py::arg("reference_dims"),
          py::arg("reference_lag"), py::arg("verbose"))
      .def("set_clip", &BatchedMocapEnv::set_clip)
      .def("set_clip_features", &BatchedMocapEnv::set_clip_features)
      .def("set_early_termination", &BatchedMocapEnv::set_early_termination)
      .def("set_init_noise", &BatchedMocapEnv::set_init_noise)
      .def("set_init_offset", &BatchedMocapEnv::set_init_offset)
      .def("set_init_state", &BatchedMocapEnv::set_init_state)
      .def("set_end_with_mocap", &BatchedMocapEnv::set_end_with_mocap)
      .def("set_reward_type", &BatchedMocapEnv::set_reward_type)
      .def("clip_ids",
           [](BatchedMocapEnv &self) { return self.clipd()->clip_ids; })
      .def("num_envs", &BatchedMocapEnv::num_envs)
      .def("action_spec", &BatchedMocapEnv::action_spec)
      .def("observation_spec", &BatchedMocapEnv::observation_spec)
      .def("reset", &BatchedMocapEnv::reset)
      .def("seed", &BatchedMocapEnv::seed)
      .def("seeds", &BatchedMocapEnv::seeds)
      .def("data",
           [](BatchedMocapEnv &self) {
             std::vector<uint64_t> ptrs;
             for (auto &it : self.data()) {
               ptrs.push_back(reinterpret_cast<uint64_t>(it));
             }
             return ptrs;
           })
      .def("model",
           [](BatchedMocapEnv &self) {
             std::vector<uint64_t> ptrs;
             for (auto &it : self.model()) {
               ptrs.push_back(reinterpret_cast<uint64_t>(it));
             }
             return ptrs;
           })
      .def("reset_if_done",
           [](BatchedMocapEnv &self) {
             py::gil_scoped_release release;
             return self.reset_if_done();
           })
      .def("step",
           [](BatchedMocapEnv &self, th::Tensor action) {
             py::gil_scoped_release release;
             return self.step(std::move(action));
           })
      .def(
          "render",
          [](BatchedMocapEnv &self, int width, int height) {
            py::gil_scoped_release release;
            return self.render_single(width, height, 0);
          },
          py::arg("width"), py::arg("height"))
      .def(
          "render_single",
          [](BatchedMocapEnv &self, int width, int height, int idx) {
            py::gil_scoped_release release;
            return self.render_single(width, height, idx);
          },
          py::arg("width"), py::arg("height"), py::arg("idx") = 0)
      .def(
          "render_all",
          [](BatchedMocapEnv &self, int width, int height) {
            py::gil_scoped_release release;
            std::vector<th::Tensor> out;
            for (auto i = 0; i < self.num_envs(); i++) {
              out.push_back(self.render_single(width, height, i));
            }
            return out;
          },
          py::arg("width"), py::arg("height"));
  m.def("quat_conj", &transforms::quat_conj);
  m.def("quat_mul", &transforms::quat_mul);
  m.def("quat_diff", &transforms::quat_diff);
  m.def("quat_inv", &transforms::quat_inv);
  m.def("quat_log", &transforms::quat_log);
  m.def("quat_dist", &transforms::quat_dist);
  m.def("bounded_quat_dist", &transforms::bounded_quat_dist);
  m.def("bounded_quat_dist2", &transforms::bounded_quat_dist2);
  m.def("quat_to_mat", &transforms::quat_to_mat);
  m.def("quat_to_axisangle", &transforms::quat_to_axisangle);
}
