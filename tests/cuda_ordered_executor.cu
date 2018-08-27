#include <executors_lite/lightweight_test.hpp>

#include <executors_lite/cuda_ordered_executor.hpp>

int main()
{
  cuda_ordered_executor oe;

  oe.execute([] __host__ __device__ { printf("1st\n"); });
  oe.execute([] __host__ __device__ { printf("2nd\n"); });
  oe.execute([] __host__ __device__ { printf("3rd\n"); });

  oe = MV(oe).then_execute([] __host__ __device__ { printf("1st\n"); });
  oe = MV(oe).then_execute([] __host__ __device__ { printf("2nd\n"); });
  oe = MV(oe).then_execute([] __host__ __device__ { printf("3rd\n"); });

  oe.wait();
}

