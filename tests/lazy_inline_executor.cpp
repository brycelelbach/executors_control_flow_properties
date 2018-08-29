#include <lightweight_test.hpp>

#include <lazy_executor.hpp>

int main()
{
  async_lazy([] { printf("1st\n"); })
    .then_execute([] { printf("2nd\n"); })
    .then_execute([] { printf("3rd\n"); })
    .flush()
  ;
}

