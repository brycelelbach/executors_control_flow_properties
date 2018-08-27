// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(EXECUTORS_LITE_CUDA_FUTURE_NEW_HPP)
#define EXECUTORS_LITE_CUDA_FUTURE_NEW_HPP

#include <executors_lite/type_deduction.hpp>

#include <executors_lite/cuda_error_handling.hpp>
#include <executors_lite/cuda_stream.hpp>
#include <executors_lite/cuda_memory.hpp>

///////////////////////////////////////////////////////////////////////////////

template <typename F>
__global__
void cuda_kernel(F f)
{
  FWD(f)();
}

template <typename Stream, typename F>
void cuda_launch_single_kernel(Stream&& stream, F&& f)
{
  cuda_kernel<<<1, 1, 0, stream.get()>>>(FWD(f));
  THROW_ON_CUDA_RT_ERROR(cudaGetLastError());
}

template <typename Stream, typename F, typename Shape, typename StateFactory>
void cuda_launch_bulk_kernel(Stream&& stream, F f, Shape n, StateFactory&& sf)
{
  int block_size;
  int min_grid_size;

  THROW_ON_CUDA_RT_ERROR(
    cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, cuda_kernel<F>, 0, 0
    )
  );

  // Compute the grid size needed based on the input size and occupancy
  // calculations, rounding up if needed.
  int const grid_size = (n + block_size - 1) / block_size;

  auto state = FWD(sf)();

  cuda_kernel<<<grid_size, block_size, 0, stream.get()>>>(
    [f, state] __device__ ()
    {
      int const idx = blockIdx.x * blockDim.x + threadIdx.x;
      f(idx, state);
    }
  );
  THROW_ON_CUDA_RT_ERROR(cudaGetLastError());
}

///////////////////////////////////////////////////////////////////////////////

struct cuda_new_stream_executor final
{
  template <typename F>
  void execute(F&& f) 
  {
    auto stream = make_cuda_unique_stream();
    cuda_launch_single_kernel(stream, FWD(f));
  }

  template <typename F, typename Shape, typename StateFactory>
  void bulk_execute(F&& f, Shape&& n, StateFactory&& sf)
  {
    auto stream = make_cuda_unique_stream();
    cuda_launch_bulk_kernel(stream, FWD(f), FWD(n), FWD(sf));
  }

  /////////////////////////////////////////////////////////////////////////////

  // No flush.

  void wait()
  {
    THROW_ON_CUDA_RT_ERROR(cudaDeviceSynchronize());
  }
};

///////////////////////////////////////////////////////////////////////////////

struct cuda_ordered_executor final
{
private:

  cuda_stream stream_; // `shared_ptr` to a CUDA stream.

public:

  cuda_ordered_executor()
    // Create a new CUDA stream.
    : stream_(make_cuda_stream())
  {}

  cuda_ordered_executor(cuda_ordered_executor const&) = default;
  cuda_ordered_executor(cuda_ordered_executor&&) = default;
  cuda_ordered_executor& operator=(cuda_ordered_executor const&) = default;
  cuda_ordered_executor& operator=(cuda_ordered_executor&& other) = default;

  /////////////////////////////////////////////////////////////////////////////

  template <typename F>
  cuda_ordered_executor then_execute(F&& f) 
  {
    cuda_launch_single_kernel(stream_, FWD(f));
    return *this;
  }

  template <typename F, typename Shape, typename StateFactory>
  cuda_ordered_executor bulk_then_execute(F&& f, Shape&& n, StateFactory&& sf)
  {
    cuda_launch_bulk_kernel(stream_, FWD(f), FWD(n), FWD(sf));
    return *this;
  }

  /////////////////////////////////////////////////////////////////////////////

  // No flush.

  void wait()
  {
    THROW_ON_CUDA_RT_ERROR(cudaStreamSynchronize(stream_.get()));
  }
};

///////////////////////////////////////////////////////////////////////////////

#endif // EXECUTORS_LITE_CUDA_FUTURE_NEW_HPP

