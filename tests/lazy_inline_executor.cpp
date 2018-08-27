#include <executors_lite/lightweight_test.hpp>

#include <executors_lite/lazy_executor.hpp>

int main()
{
  async_lazy([] { printf("1st\n"); })
    .then_execute([] { printf("2nd\n"); })
    .then_execute([] { printf("3rd\n"); })
    .flush()
  ;
}

