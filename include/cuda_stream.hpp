// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(EXECUTORS_CONTROL_FLOW_PROPERTIES_CUDA_STREAM_HPP)
#define EXECUTORS_CONTROL_FLOW_PROPERTIES_CUDA_STREAM_HPP

#include <memory>

#include <cuda_runtime.h>

#include <executors_lite/cuda_error_handling.hpp>

///////////////////////////////////////////////////////////////////////////////

struct cuda_stream_deleter final
{
  void operator()(CUstream_st* s) const
  {
    if (nullptr != s)
      THROW_ON_CUDA_RT_ERROR(cudaStreamDestroy(s));
  }
};

inline
auto make_cuda_unique_stream()
{
  CUstream_st* s;
  THROW_ON_CUDA_RT_ERROR(cudaStreamCreate(&s));
  return std::unique_ptr<CUstream_st, cuda_stream_deleter>(
    s, cuda_stream_deleter{}
  );
}

using cuda_unique_stream = decltype(make_cuda_unique_stream());

inline
auto make_cuda_shared_stream()
{
  CUstream_st* s;
  THROW_ON_CUDA_RT_ERROR(cudaStreamCreate(&s));
  return std::shared_ptr<CUstream_st>(s, cuda_stream_deleter{});
}

using cuda_shared_stream = decltype(make_cuda_shared_stream());

inline
auto make_cuda_stream()
{
  return make_cuda_shared_stream();
}

using cuda_stream = cuda_shared_stream;

///////////////////////////////////////////////////////////////////////////////

#endif // EXECUTORS_CONTROL_FLOW_PROPERTIES_CUDA_STREAM_HPP

