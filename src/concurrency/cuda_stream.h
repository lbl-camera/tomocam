
#ifndef CUDA_STREAM__H
#define CUDA_STREAM__H

#include <cuda_runtime.h>

#include "gpu/utils.cuh"

/// A simple RAII wrapper for CUDA streams.
namespace tomocam::gpu {
    class CudaStream {
      private:
        cudaStream_t stream_;

      public:
        CudaStream() { SAFE_CALL(cudaStreamCreate(&stream_)); }
        ~CudaStream() { SAFE_CALL(cudaStreamDestroy(stream_)); }
        cudaStream_t get() const { return stream_; }

        // Disable copy semantics
        CudaStream(const CudaStream &) = delete;
        CudaStream &operator=(const CudaStream &) = delete;
        // Enable move semantics
        CudaStream(CudaStream &&other) noexcept : stream_(other.stream_) {
            other.stream_ = nullptr;
        }
        CudaStream &operator=(CudaStream &&other) noexcept {
            if (this != &other) {
                SAFE_CALL(cudaStreamDestroy(stream_));
                stream_ = other.stream_;
                other.stream_ = nullptr;
            }
            return *this;
        }
    };
} // namespace tomocam::gpu

#endif // CUDA_STREAM__H
