/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef POOL_H
#define POOL_H

#include <concepts>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>

namespace tomocam::concurrency {
    template <typename T>
    concept RAIIResource =
        std::is_default_constructible_v<T> && std::is_destructible_v<T> &&
        !std::is_copy_constructible_v<T>;

    template <typename Resource, size_t Size = 5>
        requires RAIIResource<Resource>
    class Pool {
      private:
        std::mutex mtx_;
        std::condition_variable cv_;
        std::deque<std::unique_ptr<Resource>> resources_;

      public:
        Pool() = default;
        template <typename Factory, typename... Args>
        explicit Pool(const Factory &factory, Args &&...args) {
            for (size_t i = 0; i < Size; ++i) {
                resources_.push_back(std::make_unique<Resource>(
                    factory(std::forward<Args>(args)...)));
            }
        }
        ~Pool() = default;

        // Disable copy and move semantics
        Pool(const Pool &) = delete;
        Pool &operator=(const Pool &) = delete;
        Pool(Pool &&) = delete;
        Pool &operator=(Pool &&) = delete;

        void push(Resource &&resource) {
            std::unique_lock<std::mutex> lock(mtx_);
            resources_.push_back(std::make_unique<Resource>(std::move(resource)));
            lock.unlock();
            cv_.notify_one();
        }

        std::shared_ptr<Resource> acquire() {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this]() { return !resources_.empty(); });

            std::unique_ptr<Resource> resource = std::move(resources_.front());
            resources_.pop_front();

            return std::shared_ptr<Resource>(
                resource.release(), [this](Resource *res) {
                    std::unique_lock<std::mutex> lock(mtx_);
                    resources_.push_back(std::unique_ptr<Resource>(res));
                    lock.unlock();
                    cv_.notify_one();
                });
        }
    };
} // namespace tomocam::concurrency
#endif // POOL_H
