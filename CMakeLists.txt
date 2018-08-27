# Copyright (c) 2018 NVIDIA Corporation
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
  
cmake_minimum_required(VERSION 3.11)

project(EXECUTORS_LITE CXX)

find_package(CUDA VERSION 9.0)

if (CUDA_FOUND)
  enable_language(CUDA)
endif()

include(CTest)
enable_testing()
  
file(GLOB EXECUTORS_LITE_TESTS CONFIGURE_DEPENDS tests/*.cpp)

if (CUDA_FOUND)
  file(GLOB EXECUTORS_LITE_CUDA_TESTS CONFIGURE_DEPENDS tests/*.cu)
  set(EXECUTORS_LITE_TESTS "${EXECUTORS_LITE_TESTS}:${EXECUTORS_LITE_TESTS}")
endif()

foreach(source_file IN LISTS EXECUTORS_LITE_TESTS)
  get_filename_component(executable ${source_file} NAME_WE)

  add_executable(${executable} ${source_file})

  target_include_directories(
    ${executable}
    PUBLIC ${PROJECT_SOURCE_DIR}/executors_lite
    PRIVATE ${PROJECT_SOURCE_DIR}/external/executors_impl
  )

  add_test(run.${executable} ${executable})
endforeach()

