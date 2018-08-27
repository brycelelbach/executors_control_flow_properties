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
void
cuda_execute_impl(F f)
{
  FWD(f)();
}

struct cuda_ordered_executor final
{
private:

  // This is a `unique_ptr`, we can make it a `shared_ptr` to ensure it's
  // copyable if desired.
  cuda_stream stream_;

public:

  cuda_ordered_executor() = default;

  cuda_ordered_executor(cuda_ordered_executor const&) = delete;
  cuda_ordered_executor(cuda_ordered_executor&&) = default;
  cuda_ordered_executor& operator=(cuda_ordered_executor const&) = delete;
  cuda_ordered_executor& operator=(cuda_ordered_executor&& other) = default;

  /////////////////////////////////////////////////////////////////////////////
  // Internal (GPU) -> Internal (GPU) Dependent Execution.

  template <typename Invocable>
  void
  execute(Invocable&& f)
  {
    cuda_execute_impl<<<1, 1, 0, stream_.get()>>>(FWD(f));
    THROW_ON_CUDA_RT_ERROR(cudaGetLastError());
  }

  template <typename Invocable>
  cuda_ordered_executor
  then_execute(Invocable&& f) &&
  {
    execute(FWD(f));
    return MV(*this);
  }

  template <typename ElementInvocable, typename Shape, typename StateFactory>
  void
  bulk_execute(ElementInvocable ef, Shape n, StateFactory&& sf)
  {
    int block_size;
    int min_grid_size;

    THROW_ON_CUDA_RT_ERROR(
      cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, cuda_execute_impl<ElementInvocable>, 0, 0
      )
    );

    // Compute the grid size needed based on the input size and occupancy
    // calculations, rounding up if needed.
    int const grid_size = (n + block_size - 1) / block_size;

    auto state = FWD(sf)();

    cuda_execute_impl<<<grid_size, block_size, 0, stream_.get()>>>(
      [ef, state] __device__ ()
      {
        int const idx = blockIdx.x * blockDim.x + threadIdx.x;
        ef(idx, state);
      }
    );
    THROW_ON_CUDA_RT_ERROR(cudaGetLastError());
  }

  template <typename ElementInvocable, typename Shape, typename StateFactory>
  cuda_ordered_executor
  bulk_then_execute(ElementInvocable&& ef, Shape&& n, StateFactory&& sf) &&
  {
    bulk_execute(FWD(ef), FWD(n), FWD(sf));
    return MV(*this);
  }

  /////////////////////////////////////////////////////////////////////////////

  void flush() {}

  void wait()
  {
    THROW_ON_CUDA_RT_ERROR(cudaStreamSynchronize(stream_.get()));
  }
};

///////////////////////////////////////////////////////////////////////////////

#endif // EXECUTORS_LITE_CUDA_FUTURE_NEW_HPP

