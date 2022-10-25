/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mujoco.h>
#include <torch/torch.h>

namespace th = torch;
using TensorMap = std::unordered_map<std::string, th::Tensor>;

extern char const *g_warning_names[8];

struct MjDataTensors {
  th::Tensor qpos;
  th::Tensor qvel;
  th::Tensor act;
  th::Tensor subtree_com;
  th::Tensor xpos;
  th::Tensor xmat;
  th::Tensor xquat;
  th::Tensor ctrl;
  th::Tensor sensordata;
  th::Tensor cfrc_ext;
  th::Tensor qpos32;
  th::Tensor qvel32;
  th::Tensor act32;
  th::Tensor subtree_com32;
  th::Tensor xpos32;
  th::Tensor xmat32;
  th::Tensor xquat32;
  th::Tensor ctrl32;
  th::Tensor sensordata32;
  th::Tensor cfrc_ext32;

  MjDataTensors() {}
  MjDataTensors(mjModel *m, mjData *d);

  void updateFloat32();
};

struct MjRenderContext {
  mjvScene scn;
  mjvCamera cam;
  mjvOption opt;
  mjrContext con;

  MjRenderContext(mjModel *m, int cam_id = 0);
  th::Tensor render(mjModel *m, mjData *d, int w, int h);
};

void activate();
void init_opengl();

// From https://stackoverflow.com/a/27118537
class Barrier {
public:
  explicit Barrier(std::size_t size) : size_(size), count_(size), gen_(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock{mutex_};
    auto gen = gen_;
    if (!--count_) {
      gen_++;
      count_ = size_;
      cond_.notify_all();
    } else {
      cond_.wait(lock, [this, gen] { return gen != gen_; });
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable cond_;
  std::size_t size_;
  std::size_t count_;
  std::size_t gen_;
};

// From https://stackoverflow.com/a/2072890
inline bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size()) {
    return false;
  }
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
