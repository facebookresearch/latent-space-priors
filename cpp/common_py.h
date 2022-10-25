/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mocap_env.h"

th::Tensor th_from_numpy(pybind11::buffer buf);
template <typename T> std::vector<T> py_vector(pybind11::object const &obj) {
  std::vector<T> ret;
  for (auto it : obj) {
    ret.push_back(it.cast<T>());
  }
  return ret;
}

std::vector<std::string> featuresForObservable(std::string observable);

th::Tensor loadFeature(pybind11::object h5file, std::string const &id,
                       std::string const &name);

void parseClipData(
    ClipData &d, pybind11::object task,
    std::unordered_map<std::string, std::vector<std::string>> const
        &observables,
    bool verbose = false);
void parseTaskData(
    TaskData &d, pybind11::object task,
    std::unordered_map<std::string, std::vector<std::string>> const
        &observables,
    bool verbose = false);
