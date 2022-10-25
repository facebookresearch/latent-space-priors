/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>

#include <pwd.h>
#include <unistd.h>

// select EGL, OSMESA, GLFW (or disable rendering)
#if defined(MJ_EGL)
#include <EGL/egl.h>
#elif defined(MJ_OSMESA)
#include <GL/osmesa.h>
OSMesaContext ctx;
unsigned char buffer[10000000];
#elif defined(MJ_GLFW3)
#include "glfw3.h"
#endif

#include "common.h"

namespace {
void do_init_opengl() {
  //------------------------ EGL
#if defined(MJ_EGL)
  // desired config
  const EGLint configAttribs[] = {EGL_RED_SIZE,
                                  8,
                                  EGL_GREEN_SIZE,
                                  8,
                                  EGL_BLUE_SIZE,
                                  8,
                                  EGL_ALPHA_SIZE,
                                  8,
                                  EGL_DEPTH_SIZE,
                                  24,
                                  EGL_STENCIL_SIZE,
                                  8,
                                  EGL_COLOR_BUFFER_TYPE,
                                  EGL_RGB_BUFFER,
                                  EGL_SURFACE_TYPE,
                                  EGL_PBUFFER_BIT,
                                  EGL_RENDERABLE_TYPE,
                                  EGL_OPENGL_BIT,
                                  EGL_NONE};

  // get default display
  EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (eglDpy == EGL_NO_DISPLAY)
    mju_error_i("Could not get EGL display, error 0x%x\n", eglGetError());

  // initialize
  EGLint major, minor;
  if (eglInitialize(eglDpy, &major, &minor) != EGL_TRUE)
    mju_error_i("Could not initialize EGL, error 0x%x\n", eglGetError());

  // choose config
  EGLint numConfigs;
  EGLConfig eglCfg;
  if (eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs) !=
      EGL_TRUE)
    mju_error_i("Could not choose EGL config, error 0x%x\n", eglGetError());

  // bind OpenGL API
  if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE)
    mju_error_i("Could not bind EGL OpenGL API, error 0x%x\n", eglGetError());

  // create context
  EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
  if (eglCtx == EGL_NO_CONTEXT)
    mju_error_i("Could not create EGL context, error 0x%x\n", eglGetError());

  // make context current, no surface (let OpenGL handle FBO)
  if (eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx) !=
      EGL_TRUE)
    mju_error_i("Could not make EGL context current, error 0x%x\n",
                eglGetError());

    //------------------------ OSMESA
#elif defined(MJ_OSMESA)
  // create context
  ctx = OSMesaCreateContextExt(GL_RGBA, 24, 8, 8, 0);
  if (!ctx)
    mju_error("OSMesa context creation failed");

  // make current
  if (!OSMesaMakeCurrent(ctx, buffer, GL_UNSIGNED_BYTE, 800, 800))
    mju_error("OSMesa make current failed");

    //------------------------ GLFW
#elif defined(MJ_GLFW3)
  // init GLFW
  if (!glfwInit())
    mju_error("Could not initialize GLFW");

  // create invisible window, single-buffered
  glfwWindowHint(GLFW_VISIBLE, 0);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
  GLFWwindow *window =
      glfwCreateWindow(800, 800, "Invisible window", NULL, NULL);
  if (!window)
    mju_error("Could not create GLFW window");

  // make context current
  glfwMakeContextCurrent(window);
#else
  throw std::runtime_error("No rendering backend available");
#endif
}
} // namespace

char const *g_warning_names[8] = {
    "(near) singular inertia matrix",
    "too many contacts in contact list",
    "too many constraints",
    "too many visual geoms",
    "bad number in qpos",
    "bad number in qvel",
    "bad number in qacc",
    "bad number in ctrl",
};

MjDataTensors::MjDataTensors(mjModel *m, mjData *d) {
  auto dtype = th::dtype(th::kFloat64);
  qpos = th::from_blob(d->qpos, {m->nq}, dtype);
  qvel = th::from_blob(d->qvel, {m->nv}, dtype);
  act = th::from_blob(d->act, {m->na}, dtype);
  subtree_com = th::from_blob(d->subtree_com, {m->nbody, 3}, dtype);
  xpos = th::from_blob(d->xpos, {m->nbody, 3}, dtype);
  xmat = th::from_blob(d->xmat, {m->nbody, 9}, dtype);
  xquat = th::from_blob(d->xquat, {m->nbody, 4}, dtype);
  ctrl = th::from_blob(d->ctrl, {m->nu}, dtype);
  sensordata = th::from_blob(d->sensordata, {m->nsensordata}, dtype);
  cfrc_ext = th::from_blob(d->cfrc_ext, {m->nbody, 6}, dtype);
  qpos32 = qpos.to(th::kFloat32);
  qvel32 = qvel.to(th::kFloat32);
  act32 = act.to(th::kFloat32);
  subtree_com32 = subtree_com.to(th::kFloat32);
  xpos32 = xpos.to(th::kFloat32);
  xmat32 = xmat.to(th::kFloat32);
  xquat32 = xquat.to(th::kFloat32);
  ctrl32 = ctrl.to(th::kFloat32);
  sensordata32 = sensordata.to(th::kFloat32);
  cfrc_ext32 = cfrc_ext.to(th::kFloat32);
}

void MjDataTensors::updateFloat32() {
  qpos32.copy_(qpos);
  qvel32.copy_(qvel);
  act32.copy_(act);
  subtree_com32.copy_(subtree_com);
  xpos32.copy_(xpos);
  xmat32.copy_(xmat);
  xquat32.copy_(xquat);
  ctrl32.copy_(ctrl);
  sensordata32.copy_(sensordata);
  cfrc_ext32.copy_(cfrc_ext);
}

MjRenderContext::MjRenderContext(mjModel *m, int cam_id) {
  init_opengl();

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, 200);

  cam.fixedcamid = cam_id;
  cam.type = mjCAMERA_FIXED;
}

th::Tensor MjRenderContext::render(mjModel *m, mjData *d, int w, int h) {
  // set rendering to offscreen buffer
  mjr_setBuffer(mjFB_OFFSCREEN, &con);
  if (con.currentBuffer != mjFB_OFFSCREEN) {
    std::cerr << "Warning: offscreen rendering not supported, using "
                 "default/window framebuffer"
              << std::endl;
  }

  // get size of active renderbuffer
  mjrRect viewport = mjr_maxViewport(&con);
  int W = std::min(w, viewport.width);
  int H = std::min(h, viewport.height);
  viewport.width = W;
  viewport.height = H;

  auto rgb = th::empty({H, W, 3}, th::kUInt8);

  // update abstract scene
  mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);

  // render scene in offscreen buffer
  mjr_render(viewport, &scn, &con);

  // read rgb and depth buffers
  mjr_readPixels(static_cast<uint8_t *>(rgb.data_ptr()), nullptr, viewport,
                 &con);

  return rgb.flip({0});
}

void activate() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    char const *homedir;
    if ((homedir = getenv("HOME")) == NULL) {
      homedir = getpwuid(getuid())->pw_dir;
    }
    auto mjkey = std::string(homedir) + "/.mujoco/mjkey.txt";
    mj_activate(mjkey.c_str());
  });
}

void init_opengl() {
  static std::once_flag flag;
  std::call_once(flag, [] { do_init_opengl(); });
}
