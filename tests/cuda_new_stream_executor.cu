#include <lightweight_test.hpp>

#include <cuda_executor.hpp>

int main()
{
  cuda_new_stream_executor nse;

  nse.execute([] __host__ __device__ { printf("1st\n"); });
  nse.execute([] __host__ __device__ { printf("2nd\n"); });
  nse.execute([] __host__ __device__ { printf("3rd\n"); });

  nse.wait(); 
}

