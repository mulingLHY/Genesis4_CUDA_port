#ifndef GENESIS4_CUDA_LAUNCH_H_
#define GENESIS4_CUDA_LAUNCH_H_

#ifdef GENESIS_USE_CUDA

#include "Genesis4CudaRuntime.h"

#define GENESIS4_CUDA_DEVICE __device__
#define GENESIS4_CUDA_HOST_DEVICE __host__ __device__
#define GENESIS4_CUDA_FORCE_INLINE __forceinline__


GENESIS4_CUDA_DEVICE GENESIS4_CUDA_FORCE_INLINE
double g4_cuda_atomic_add_double(double* address, double value) noexcept
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#else
  return atomicAdd(address, value);
#endif
}

template <typename F>
__global__ void g4_parallel_for_kernel(long long n, F f)
{
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (; i < n; i += stride) {
    f(i);
  }
}

template <typename F>
inline void g4_parallel_for(long long n, F f, cudaStream_t stream = g4_cuda_stream())
{
  if (n <= 0) { return; }
  constexpr int block = 256;
  long long grid_ll = (n + block - 1) / block;
  if (grid_ll > 2147483647LL) { grid_ll = 2147483647LL; }
  const int grid = static_cast<int>(grid_ll);
  g4_parallel_for_kernel<<<grid, block, 0, stream>>>(n, f);
  g4_cuda_check(cudaGetLastError(), "g4_parallel_for launch");
}

#endif // GENESIS_USE_CUDA

#endif // GENESIS4_CUDA_LAUNCH_H_
