/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef GPUTOHOST__H
#define GPUTOHOST__H

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>

#include "gpu/utils.cuh"
#include "machine.h"

namespace tomocam {
    template <typename Host_t, typename Device_t>
    class GPUToHost {
      private:
        std::thread thread_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<std::tuple<Host_t, Device_t>> queue_;
        bool stop_;

      public:
        GPUToHost() : stop_(false) {
            // CUDA's "current device" is thread-local and is NOT inherited by a
            // newly spawned thread (a fresh thread always starts on device 0).
            // Capture the calling thread's device and re-select it here, so the
            // device-to-host cudaMemcpy -- and the cudaFree that runs when the
            // Device_t below is destroyed -- are issued on the device that
            // actually owns the memory, not silently on device 0.
            int device = 0;
            SAFE_CALL(cudaGetDevice(&device));
            thread_ = std::thread([this, device] {
                DeviceGuard guard(device);
                while (true) {
                    auto item = pop();
                    if (!item.has_value()) break;
                    auto &&[h, d] = std::move(item.value());
                    d.copy_to(h);
                }
            });
        }

        ~GPUToHost() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            cv_.notify_all();
            thread_.join();
        }

        void push(Host_t h, Device_t &&d) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(std::make_tuple(h, std::move(d)));
            }
            cv_.notify_one();
        }

      private:
        GPUToHost(const GPUToHost &) = delete;
        GPUToHost &operator=(const GPUToHost &) = delete;
        GPUToHost(GPUToHost &&) = delete;
        GPUToHost &operator=(GPUToHost &&) = delete;

        // blocks until an item is available, or returns nullopt once the queue
        // has been drained *and* the shipper has been told to stop
        std::optional<std::tuple<Host_t, Device_t>> pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || stop_; });
            if (queue_.empty()) return std::nullopt;
            auto item = std::move(queue_.front());
            queue_.pop();
            return item;
        }
    };
} // namespace tomocam

#endif // GPUTOHOST__H
