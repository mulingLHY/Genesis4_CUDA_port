#ifndef GENESIS4_CUDA_BUFFER_H_
#define GENESIS4_CUDA_BUFFER_H_

#ifdef GENESIS_USE_CUDA

#include "Genesis4CudaRuntime.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace genesis4_cuda {

template <typename T>
class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;
  ~CudaDeviceBuffer() { clear(); }

  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

  CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), capacity_(other.capacity_)
  {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept
  {
    if (this != &other) {
      clear();
      ptr_ = other.ptr_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.ptr_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  void resize(std::size_t n)
  {
    if (n <= capacity_) {
      size_ = n;
      return;
    }

    T* new_ptr = nullptr;
    if (n > 0) {
      void* raw_ptr = nullptr;
      check(cudaMalloc(&raw_ptr, n * sizeof(T)), "cudaMalloc CudaDeviceBuffer");
      new_ptr = static_cast<T*>(raw_ptr);
      if (ptr_ != nullptr && size_ > 0) {
        const std::size_t ncopy = std::min(size_, n);
        check(cudaMemcpyAsync(new_ptr, ptr_, ncopy * sizeof(T), cudaMemcpyDeviceToDevice, stream()),
              "cudaMemcpyAsync CudaDeviceBuffer grow");
        check(cudaStreamSynchronize(stream()), "cudaStreamSynchronize CudaDeviceBuffer grow");
      }
    }

    if (ptr_ != nullptr) {
      check(cudaFree(ptr_), "cudaFree CudaDeviceBuffer grow");
    }
    ptr_ = new_ptr;
    size_ = n;
    capacity_ = n;
  }

  void clear()
  {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = 0;
    capacity_ = 0;
  }

  void copy_from_host(const T* src, std::size_t n)
  {
    resize(n);
    if (n > 0) {
      check(cudaMemcpyAsync(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice, stream()),
            "cudaMemcpyAsync host to device");
    }
  }

  void copy_to_host(T* dst, std::size_t n) const
  {
    if (n > size_) {
      throw std::runtime_error("CudaDeviceBuffer::copy_to_host requested more elements than available");
    }
    if (n > 0) {
      check(cudaMemcpyAsync(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, stream()),
            "cudaMemcpyAsync device to host");
    }
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  T* begin() { return ptr_; }
  const T* begin() const { return ptr_; }
  T* end() { return ptr_ + size_; }
  const T* end() const { return ptr_ + size_; }
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

template <typename T>
class CudaPinnedBuffer {
 public:
  CudaPinnedBuffer() = default;
  ~CudaPinnedBuffer() { clear(); }

  CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
  CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;

  void resize(std::size_t n)
  {
    if (n <= capacity_) {
      size_ = n;
      return;
    }
    clear();
    if (n > 0) {
      void* raw_ptr = nullptr;
      check(cudaMallocHost(&raw_ptr, n * sizeof(T)), "cudaMallocHost CudaPinnedBuffer");
      ptr_ = static_cast<T*>(raw_ptr);
    }
    size_ = n;
    capacity_ = n;
  }

  void clear()
  {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
    }
    size_ = 0;
    capacity_ = 0;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  T* begin() { return ptr_; }
  const T* begin() const { return ptr_; }
  T* end() { return ptr_ + size_; }
  const T* end() const { return ptr_ + size_; }
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }
  T& operator[](std::size_t i) { return ptr_[i]; }
  const T& operator[](std::size_t i) const { return ptr_[i]; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

template <typename SrcIter, typename DstIter>
inline void copy_host_to_device(SrcIter first, SrcIter last, DstIter dst)
{
  const std::size_t n = static_cast<std::size_t>(last - first);
  if (n > 0) {
    using Value = typename std::iterator_traits<SrcIter>::value_type;
    const Value* src_ptr = &(*first);
    Value* dst_ptr = &(*dst);
    check(cudaMemcpyAsync(dst_ptr, src_ptr, n * sizeof(Value), cudaMemcpyHostToDevice, stream()),
          "cudaMemcpyAsync host to device");
  }
}

template <typename SrcIter, typename DstIter>
inline void copy_device_to_host(SrcIter first, SrcIter last, DstIter dst)
{
  const std::size_t n = static_cast<std::size_t>(last - first);
  if (n > 0) {
    using Value = typename std::iterator_traits<SrcIter>::value_type;
    const Value* src_ptr = &(*first);
    Value* dst_ptr = &(*dst);
    check(cudaMemcpyAsync(dst_ptr, src_ptr, n * sizeof(Value), cudaMemcpyDeviceToHost, stream()),
          "cudaMemcpyAsync device to host");
  }
}

} // namespace genesis4_cuda

using genesis4_cuda::CudaDeviceBuffer;
using genesis4_cuda::CudaPinnedBuffer;

#endif // GENESIS_USE_CUDA

#endif // GENESIS4_CUDA_BUFFER_H_
