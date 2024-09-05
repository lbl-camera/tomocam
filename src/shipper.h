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

#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>

namespace tomocam {
    template <typename Host_t, typename Device_t>
    class GPUToHost {
      private:
        std::thread thread_;
        std::mutex mutex_;
        std::queue<std::tuple<Host_t, Device_t>> queue_;
        std::atomic<bool> stop_;

      public:
        GPUToHost() : stop_(false) {
            thread_ = std::thread([this] {
                while (!queue_.empty() || !stop_) {
                    auto item = pop();
                    if (item.has_value()) {
                        auto [h, d] = item.value();
                        d.copy_to(h);
                    }
                }
            });
        }

        ~GPUToHost() {
            stop_ = true;
            thread_.join();
        }

        void push(Host_t h, Device_t d) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::make_tuple(h, d));
        }

      private:
        GPUToHost(const GPUToHost &) = delete;
        GPUToHost &operator=(const GPUToHost &) = delete;
        GPUToHost(GPUToHost &&) = delete;
        GPUToHost &operator=(GPUToHost &&) = delete;

        std::optional<std::tuple<Host_t, Device_t>> pop() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) return std::nullopt;
            auto item = queue_.front();
            queue_.pop();
            return item;
        }
    };
} // namespace tomocam

#endif // GPUTOHOST__H
