cmake_minimum_required(VERSION 3.14)

#project(pytorch_cmake_example LANGUAGES CXX CUDA)
project(hucc_cpp LANGUAGES CXX)

execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import torch;print(int(torch.cuda.is_available()))" OUTPUT_VARIABLE HAVE_CUDA OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PATH}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Get relevant PyTorch compile flags the lazy and hacky way
execute_process(COMMAND python -c "import torch._C;print(f'-DYBIND11_COMPILER_TYPE=\"{torch._C._PYBIND11_COMPILER_TYPE}\"'); print(f'-DYBIND11_STDLIB=\"{torch._C._PYBIND11_STDLIB}\"'); print(f'-DYBIND11_BUILD_ABI=\"{torch._C._PYBIND11_BUILD_ABI}\"');print(f'-D_GLIBCXX_USE_CXX11_ABI={torch.compiled_with_cxx11_abi():d}')" OUTPUT_VARIABLE TORCH_EXT_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" TORCH_EXT_FLAGS_LIST ${TORCH_EXT_FLAGS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(PYBIND11_FINDPYTHON ON)
set(Python_FIND_STRATEGY LOCATION)
add_compile_options(-Wall)# -fno-omit-frame-pointer)
find_package(pybind11 CONFIG)
find_package(Torch REQUIRED)
find_package(MUJOCO REQUIRED)
find_package(OpenGL COMPONENTS OpenGL EGL)

add_library(hucc_cpp SHARED
  mocap_env.cpp
  common.cpp
)
target_include_directories(hucc_cpp PUBLIC
  ${MUJOCO_INCLUDE_DIRS}
)
target_link_libraries(hucc_cpp PUBLIC
  ${TORCH_LIBRARIES}
  ${MUJOCO_LIBRARIES}
)

if(${HAVE_CUDA} EQUAL "1")
  target_compile_definitions(hucc_cpp PUBLIC
    -DHAVE_CUDA
  )
endif()
if(${OpenGL_EGL_FOUND}) 
  find_package(GLEW REQUIRED)
  target_compile_definitions(hucc_cpp PUBLIC
    -DMJ_EGL
  )
  target_include_directories(hucc_cpp PRIVATE
    ${OPENGL_INCLUDE_DIR}
    ${OPENGL_EGL_INCLUDE_DIRS}
  )
  target_link_libraries(hucc_cpp PUBLIC
    OpenGL::EGL
    OpenGL::GL
    GLEW::GLEW
  )
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  if (EXISTS ${MUJOCO_RUNTIME_LIBRARY_DIRS}/libglfw.3.dylib)
    target_compile_definitions(hucc_cpp PUBLIC
      -DMJ_GLFW3
    )
    target_link_libraries(hucc_cpp PUBLIC
      ${MUJOCO_RUNTIME_LIBRARY_DIRS}/libglfw.3.dylib
    )
  else()
    message(WARNING "GLFW3 not found, rendering won't work")
  endif()
else()
  target_compile_definitions(hucc_cpp PUBLIC
    -DMJ_GLFW3
  )
endif()

add_executable(mocap_test
  mocap_test.cpp
  common_py.cpp
)
target_link_libraries(mocap_test PRIVATE
  hucc_cpp
  pybind11::embed
)

Python_add_library(hucc_cpp_ext
  mocap_env_ext.cpp
  common_py.cpp
)
target_compile_definitions(hucc_cpp_ext PRIVATE
  -DTORCH_API_INCLUDE_EXTENSION_H
  -DTORCH_EXTENSION_NAME=hucc_cpp_ext
  ${TORCH_EXT_FLAGS_LIST}
)
target_link_libraries(hucc_cpp_ext PRIVATE
  hucc_cpp
  ${TORCH_INSTALL_PREFIX}/lib/libtorch_python${CMAKE_SHARED_LIBRARY_SUFFIX}
  pybind11::headers
)
set_target_properties(hucc_cpp_ext PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION ON
  CXX_VISIBILITY_PRESET default
  VISIBLITY_INLINES_HIDDEN ON
)
# Meh
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_custom_command(TARGET hucc_cpp
      POST_BUILD COMMAND
      ${CMAKE_INSTALL_NAME_TOOL} -change "@executable_path/libmujoco200.dylib" "${MUJOCO_LIBRARIES}"
      $<TARGET_FILE:hucc_cpp>)
  add_custom_command(TARGET hucc_cpp_ext
      POST_BUILD COMMAND
      ${CMAKE_INSTALL_NAME_TOOL} -change "@executable_path/libmujoco200.dylib" "${MUJOCO_LIBRARIES}"
      $<TARGET_FILE:hucc_cpp_ext>)
endif()
