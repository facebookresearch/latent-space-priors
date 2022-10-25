# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

execute_process(COMMAND python -c "import os; import mujoco; print(os.path.dirname(mujoco.__file__))" OUTPUT_VARIABLE MUJOCO_MODULE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
find_path(MUJOCO_INCLUDE_DIR mujoco.h HINTS env cplus_include_path ${MUJOCO_MODULE_PATH}/include)
find_library(MUJOCO_LIBRARY NAMES libmujoco.so HINTS env library_path env ld_library_path ${MUJOCO_MODULE_PATH})

set(MUJOCO_LIBRARIES ${MUJOCO_LIBRARY})
set(MUJOCO_INCLUDE_DIRS ${MUJOCO_INCLUDE_DIR} ${MUJOCO_INCLUDE_DIR}/mujoco)
get_filename_component(MUJOCO_BINDIR ${MUJOCO_LIBRARY} DIRECTORY)
set(MUJOCO_RUNTIME_LIBRARY_DIRS ${MUJOCO_BINDIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUJOCO DEFAULT_MSG MUJOCO_LIBRARY MUJOCO_INCLUDE_DIR MUJOCO_RUNTIME_LIBRARY_DIRS)
