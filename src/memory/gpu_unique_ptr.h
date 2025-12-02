/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef TOMOCAM_GPU_UNIQUE_PTR__H
#define TOMOCAM_GPU_UNIQUE_PTR__H

#include <cuda_runtime.h>
#include <memory>

#include "concurrency/cuda_stream.h"
#include "concurrency/global_pools.h"
#include "gpu/utils.cuh"
#include "memory/pinned_buffer.h"
#include "concurrency/pool.h"

namespace tomocam::memory {

    struct cudaDeleter {
        void operator()(void *ptr) const {
            if (ptr) SAFE_CALL(cudaFree(ptr));
        }
    };

    template <class T>
    using cuniquePtr = std::unique_ptr<T, cudaDeleter>;

    template <class T>
    cuniquePtr<T> make_cuniquePtr(size_t num_elems) {
        T *ptr = nullptr;
        if (num_elems > 0) {
            SAFE_CALL(cudaMalloc((void **)&ptr, sizeof(T) * num_elems));
        }
        return cuniquePtr<T>(ptr);
    }

    struct AsyncMemTransfer {
      public:
        static void H2D(void *d_dst, const void *h_src, size_t num_bytes) {
            if (num_bytes == 0) return;
            int device = 0;
            SAFE_CALL(cudaGetDevice(&device));

            auto stream = tomocam::global::stream_pools[device].acquire();
            auto buffer = tomocam::global::pinned_buffer_pools[device].acquire();

            if (buffer->bytes() < num_bytes) {
                throw std::runtime_error("Pinned buffer too small for H2D transfer");
            }

            std::memcpy((void *)buffer->get(), h_src, num_bytes);
            SAFE_CALL(cudaMemcpyAsync(d_dst, buffer->get(), num_bytes,
                                      cudaMemcpyHostToDevice, stream->get()));
        }

        static void D2H(void *h_dst, const void *d_src, size_t num_bytes) {
            if (num_bytes == 0) return;

            int device = 0;
            SAFE_CALL(cudaGetDevice(&device));

            auto stream = tomocam::global::stream_pools[device].acquire();
            auto buffer = tomocam::global::pinned_buffer_pools[device].acquire();

            if (buffer->bytes() < num_bytes) {
                throw std::runtime_error("Pinned buffer too small for D2H transfer");
            }

            SAFE_CALL(cudaMemcpyAsync(buffer->get(), d_src, num_bytes,
                                      cudaMemcpyDeviceToHost, stream->get()));
            SAFE_CALL(cudaStreamSynchronize(stream->get()));
            std::memcpy(h_dst, buffer->get(), num_bytes);
        }
    };

    inline void H2D(void *d_dst, const void *h_src, size_t num_bytes) {
        AsyncMemTransfer::H2D(d_dst, h_src, num_bytes);
    }

    inline void D2H(void *h_dst, const void *d_src, size_t num_bytes) {
        AsyncMemTransfer::D2H(h_dst, d_src, num_bytes);
    }
} // namespace tomocam::memory

#endif // TOMOCAM_GPU_UNIQUE_PTR__H
