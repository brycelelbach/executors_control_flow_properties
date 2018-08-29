#pragma once

#include <experimental/execution>
#include <utility>
#include <functional>

template<class OneWayExecutor>
class lazy_oneway_executor_adaptor
{
  public:
    explicit lazy_oneway_executor_adaptor(const OneWayExecutor& adapted_ex) noexcept
      : queued_work_(nullptr),
        adapted_ex_(adapted_ex)
    {}

    template<class Function>
    void execute(Function&& f) const
    {
      queued_work_ = [previous_work = std::move(queued_work_), f = std::forward<Function>(f), ex = adapted_ex_]
      {
        if(previous_work) previous_work();

        ex.execute(std::move(f));
      };
    }

    void flush() const
    {
      // run the work
      queued_work_();

      // reset the work
      queued_work_ = nullptr;
    }

  private:
    mutable std::function<void(void)> queued_work_;
    OneWayExecutor adapted_ex_;
};

