// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(EXECUTORS_LITE_INLINE_EXECUTOR_HPP)
#define EXECUTORS_LITE_INLINE_EXECUTOR_HPP

#include <executors_lite/type_deduction.hpp>

///////////////////////////////////////////////////////////////////////////////

struct inline_executor final
{
  template <typename F>
  void execute(F&& f) const
  {
    FWD(f)();
  }

  /////////////////////////////////////////////////////////////////////////////

  void wait() {}

  void flush() {}
};

///////////////////////////////////////////////////////////////////////////////

#endif // EXECUTORS_LITE_INLINE_EXECUTOR_HPP

