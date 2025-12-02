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

#ifndef TOMOCAM_PINNED_BUFFER__H
#define TOMOCAM_PINNED_BUFFER__H

#include <array>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace tomocam::memory {

    /* pinned pointer type */
    struct PinnedDeleter {
        void operator()(void *p) {
            if (p) { cudaFreeHost(p); }
        }
    };
    using pinnedPtr = std::unique_ptr<void, PinnedDeleter>;

    /* create pinned pointer */
    inline pinnedPtr make_pinnedPtr(const size_t &num_bytes) {
        void *raw_ptr = nullptr;
        const cudaError_t err = cudaMallocHost(&raw_ptr, num_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocHost failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return pinnedPtr(raw_ptr);
    }

    /* pinned buffer handle */
    class PinnedBuffer {
      private:
        pinnedPtr buffer_;
        size_t bytes_;

      public:
        PinnedBuffer() : buffer_(nullptr), bytes_(0) {}

        ~PinnedBuffer() = default;

        // Prevent copying
        PinnedBuffer(const PinnedBuffer &) = delete;
        PinnedBuffer &operator=(const PinnedBuffer &) = delete;
        // Allow moving
        PinnedBuffer(PinnedBuffer &&) = default;
        PinnedBuffer &operator=(PinnedBuffer &&) = default;

        void *get() { return buffer_.get(); }

        const void *get() const { return buffer_.get(); }
        size_t bytes() const { return bytes_; }

        PinnedBuffer static create(const size_t &num_bytes) {
            PinnedBuffer pb;
            pb.buffer_ = make_pinnedPtr(num_bytes);
            pb.bytes_ = num_bytes;
            return pb;
        }
    };
} // namespace tomocam::memory

#endif // TOMOCAM_PINNED_BUFFER__H
