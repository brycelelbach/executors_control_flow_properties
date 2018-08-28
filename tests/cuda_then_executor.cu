#include <executors_lite/cuda_then_executor.hpp>
#include <iostream>

int main()
{
  cuda_then_executor ex_a;
  cuda_then_executor ex_b;

  ex_a.then_execute([] __device__ 
  {
    printf("task a\n");
  });

  ex_b.then_execute([] __device__
  {
    printf("task b\n");
  });

  // make all tasks executed through ex_c depend on the completion of all tasks launched through ex_a & ex_b
  cuda_then_executor ex_c = cuda_then_executor().require(depends_on(ex_a, ex_b));

  ex_c.then_execute([] __device__
  {
    printf("task c dependent on tasks a & b\n");
  });

  ex_c.wait();

  std::cout << "OK" << std::endl;
}

