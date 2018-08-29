#pragma once

#include <thread>

class new_thread_executor
{
  public:
    template<class Function>
    void execute(Function&& f) const
    {
      std::thread(std::forward<Function>(f)).detach();
    }

    bool operator==(const new_thread_executor&) const noexcept { return true; }
    bool operator!=(const new_thread_executor&) const noexcept { return false; }
};

