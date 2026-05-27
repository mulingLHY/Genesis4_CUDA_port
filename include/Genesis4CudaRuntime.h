#ifndef GENESIS4_CUDA_RUNTIME_H_
#define GENESIS4_CUDA_RUNTIME_H_

#ifdef GENESIS_USE_CUDA

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace genesis4_cuda {

inline void check(cudaError_t result, const char* where)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(result));
  }
}

[[noreturn]] inline void abort(const std::string& message)
{
  throw std::runtime_error(message);
}

inline cudaStream_t stream()
{
  return cudaStream_t{0};
}

inline void synchronize()
{
  check(cudaStreamSynchronize(stream()), "cudaStreamSynchronize");
}

inline void initialize()
{
  check(cudaFree(nullptr), "cuda runtime initialization");
}

inline void finalize()
{
  synchronize();
}

inline int device_id()
{
  int dev = 0;
  check(cudaGetDevice(&dev), "cudaGetDevice");
  return dev;
}

} // namespace genesis4_cuda

inline cudaStream_t g4_cuda_stream()
{
  return genesis4_cuda::stream();
}

inline void g4_cuda_synchronize()
{
  genesis4_cuda::synchronize();
}

inline void g4_cuda_check(cudaError_t result, const char* where)
{
  genesis4_cuda::check(result, where);
}

[[noreturn]] inline void g4_cuda_abort(const std::string& message)
{
  genesis4_cuda::abort(message);
}

#endif // GENESIS_USE_CUDA

#endif // GENESIS4_CUDA_RUNTIME_H_
