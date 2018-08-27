# Copyright (c) 2018 NVIDIA Corporation
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
  
cmake_minimum_required(VERSION 3.12)

project(EXECUTORS_LITE CXX)

include(CheckLanguage)

check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 14)

include(CTest)
enable_testing()
  
file(
  GLOB EXECUTORS_LITE_TESTS CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/tests/*.cpp
)

if (CMAKE_CUDA_COMPILER)
  file(
    GLOB EXECUTORS_LITE_CUDA_TESTS CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/tests/*.cu
  )
  list(APPEND EXECUTORS_LITE_TESTS ${EXECUTORS_LITE_CUDA_TESTS})
endif()

foreach(source_file IN LISTS EXECUTORS_LITE_TESTS)
  get_filename_component(executable ${source_file} NAME_WE)

  add_executable(${executable} ${source_file})

  target_include_directories(
    ${executable}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/external/executors_impl
  )

  # CMake's CUDA support doesn't seem to recognize CMAKE_CXX_STANDARD, so we
  # need to specify it here.
  target_compile_options(
    ${executable}
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -std=c++14>
  )

  add_test(run.${executable} ${executable})
endforeach()

