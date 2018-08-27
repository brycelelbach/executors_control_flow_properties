// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(EXECUTORS_LITE_LAZY_EXECUTOR_HPP)
#define EXECUTORS_LITE_LAZY_EXECUTOR_HPP

#include <type_traits>

#include <executors_lite/type_deduction.hpp>
#include <executors_lite/inline_executor.hpp>

///////////////////////////////////////////////////////////////////////////////

template <typename F, typename Predecessor >
struct lazy_inline_executor final
{
  using this_type = lazy_inline_executor<F, Predecessor>;

  F f_;
  Predecessor pred_;

  /////////////////////////////////////////////////////////////////////////////

  template <typename G>
  auto then_execute(G&& g) &&
  {
    return lazy_inline_executor<G, this_type>{FWD(g), MV(*this)};
  }

  /////////////////////////////////////////////////////////////////////////////

  void flush() &&
  {
    MV(pred_).flush();
    MV(f_)();
  }

  void wait() &&
  {
    flush();
  }
};

template <typename G>
auto async_lazy(G&& g)
{
  return lazy_inline_executor<
    std::remove_const_t<std::remove_reference_t<G>>
  , inline_executor
  >{
    FWD(g), inline_executor{}
  };
}

///////////////////////////////////////////////////////////////////////////////

#endif // EXECUTORS_LITE_LAZY_EXECUTOR_HPP

