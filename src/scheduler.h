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

#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>
#include <queue>
#include <condition_variable>

#ifndef SCHEDULER_H
#define SCHEDULER_H

namespace tomocam {

    constexpr unsigned MAX_QUEUE_SIZE = 3;

    template <typename Host_t, typename Device_t, typename... Types>
    class Scheduler {
      private:
        std::queue<std::tuple<int, Device_t, Types...>> pending_work_;
        std::mutex m_;
        std::condition_variable cv_;
        bool all_done_;

      public:
        Scheduler() : all_done_(false) {}
        Scheduler(std::vector<Host_t> h_arr) : all_done_(false) {
            enqueue(h_arr);
        }
        Scheduler(std::vector<Host_t> h_arr1, std::vector<Host_t> h_arr2) :
            all_done_(false) {
            enqueue(h_arr1, h_arr2);
        }

        // destructor
        ~Scheduler() = default;

        // delete copy and move constructors
        Scheduler(const Scheduler &) = delete;
        Scheduler &operator=(const Scheduler &) = delete;
        Scheduler(Scheduler &&) = delete;
        Scheduler &operator=(Scheduler &&) = delete;

        // get work from queue
        std::optional<std::tuple<int, Device_t, Types...>> get_work() {
            std::lock_guard<std::mutex> lock(m_);
            if (pending_work_.empty()) {
                cv_.notify_one();
                return std::nullopt;
            }
            auto work = pending_work_.front();
            pending_work_.pop();
            cv_.notify_one();
            return work;
        }

        // check if queue has work
        bool has_work() const { return (!all_done_ || !pending_work_.empty()); }

      private:
        // enqueue one std::vector of Host_t
        void enqueue(std::vector<Host_t> h_arr) {
            std::thread([this, h_arr]() {
                for (int i = 0; i < h_arr.size(); i++) {
                    std::unique_lock<std::mutex> lock(this->m_);
                    cv_.wait(lock, [this]() {
                        return this->pending_work_.size() < MAX_QUEUE_SIZE;
                    });
                    Device_t d_arr(h_arr[i]);
                    this->pending_work_.push(std::make_tuple(i, d_arr));
                }
                this->all_done_ = true;
            }).detach();
        }

        // enqueue two std::vector of Host_t
        void enqueue(std::vector<Host_t> h_arr1, std::vector<Host_t> h_arr2) {
            std::thread([this, h_arr1, h_arr2]() {
                for (int i = 0; i < h_arr1.size(); i++) {
                    std::unique_lock<std::mutex> lock(this->m_);
                    cv_.wait(lock, [this]() {
                        return this->pending_work_.size() < MAX_QUEUE_SIZE;
                    });
                    Device_t d_arr1(h_arr1[i]);
                    Device_t d_arr2(h_arr2[i]);
                    this->pending_work_.push(
                        std::make_tuple(i, d_arr1, d_arr2));
                }
                this->all_done_ = true;
            }).detach();
        }
    };

} // namespace tomocam

#endif // SCHEDULER_H
