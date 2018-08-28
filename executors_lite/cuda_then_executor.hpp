#pragma once

#include <stdexcept>
#include <utility>
#include <type_traits>
#include <iostream>
#include "depends_on.hpp"

#define __REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

namespace detail
{


template<class... Conditions>
struct conjunction;

template<>
struct conjunction<> : std::true_type {};

template<class Condition, class... Conditions>
struct conjunction<Condition, Conditions...>
  : std::integral_constant<
      bool,
      Condition::value && conjunction<Conditions...>::value
    >
{};


template<class Function>
__global__ void kernel(Function f)
{
  f();
}


} // end detail


class cuda_then_executor
{
  public:
    cuda_then_executor()
      : stream_(make_cuda_stream())
    {}

    cuda_then_executor(const cuda_then_executor& other)
      : cuda_then_executor()
    {
      depend_on(stream_, other.stream_);
    }

    cuda_then_executor(cuda_then_executor&& other)
      : stream_{}
    {
      std::swap(stream_, other.stream_);
    }

    ~cuda_then_executor()
    {
      destroy_cuda_stream(stream_);
    }

    template<class F>
    cuda_then_executor then_execute(F f) const
    {
      detail::kernel<<<1,1,0,stream_>>>(f);

      // return a copy of *this
      return *this;
    }

    void wait() const
    {
      if(auto error = cudaStreamSynchronize(stream_))
      {
        throw std::runtime_error("CUDA error after cudaStreamSynchronize(): " + std::string(cudaGetErrorString(error)));
      }
    }

    template<class... CudaThenExecutors,
             __REQUIRES(
               detail::conjunction<
                 std::is_same<cuda_then_executor, CudaThenExecutors>...
               >::value
             )>
    cuda_then_executor require(const depends_on_t<CudaThenExecutors...>& dependencies) const
    {
      return cuda_then_executor(dependencies.value());
    }

    std::tuple<cuda_then_executor> query(const depends_on_t<void>) const
    {
      return std::make_tuple(*this);
    }

  private:
    template<class... CudaThenExecutors>
    cuda_then_executor(const std::tuple<CudaThenExecutors...>& dependencies)
      : cuda_then_executor()
    {
      // make stream_ depend on each of streams
      depend_on_all(std::make_index_sequence<sizeof...(CudaThenExecutors)>(), stream_, dependencies);
    }

    static cudaStream_t make_cuda_stream()
    {
      cudaStream_t result{};
      if(auto error = cudaStreamCreate(&result))
      {
        throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
      }

      return result;
    }

    static void destroy_cuda_stream(cudaStream_t s)
    {
      if(auto error = cudaStreamDestroy(s))
      {
        std::cerr << "CUDA error after cudaStreamDestroy(): " << cudaGetErrorString(error);
        std::terminate();
      }
    }

    // this function returns an int so it can be unpacked into the swallow(...) function
    static int depend_on(cudaStream_t s, cudaStream_t dependency)
    {
      // record an event on dependency
      cudaEvent_t event{};
      if(auto error = cudaEventCreate(&event, cudaEventDisableTiming))
      {
        throw std::runtime_error("CUDA error after cudaEventCreate(): " + std::string(cudaGetErrorString(error)));
      }

      if(auto error = cudaEventRecord(event, dependency))
      {
        throw std::runtime_error("CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
      }

      // make s wait on the event
      if(auto error = cudaStreamWaitEvent(s, event, 0))
      {
        throw std::runtime_error("CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
      }

      // destroy the event
      if(auto error = cudaEventDestroy(event))
      {
        throw std::runtime_error("CUDA error after cudaEventDestroy(): " + std::string(cudaGetErrorString(error)));
      }

      return 0;
    }

    template<class... Args>
    static void swallow(Args&&...) {}

    template<std::size_t... Indices, class TupleOfCudaThenExecutors>
    static void depend_on_all(std::index_sequence<Indices...>, cudaStream_t s, const TupleOfCudaThenExecutors& cuda_then_executors)
    {
      swallow(depend_on(s, std::get<Indices>(cuda_then_executors).stream_)...);
    }

    cudaStream_t stream_;
};

