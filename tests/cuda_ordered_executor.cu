#include <executors_lite/lightweight_test.hpp>

#include <executors_lite/cuda_executor.hpp>

int main()
{
  cuda_ordered_executor oe;

  oe
    .then_execute([] __host__ __device__ { printf("1st\n"); })
    .then_execute([] __host__ __device__ { printf("2nd\n"); })
    .then_execute([] __host__ __device__ { printf("3rd\n"); })
  ;

  oe.wait();
}

