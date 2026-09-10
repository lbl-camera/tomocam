
#include "dist_array.h"
#include "gpu/gpu_ops.cuh"
#include "scheduler.h"

namespace tomocam::array {
    template <typename T>
    T dot(const Partition<T> &p1, const Partition<T> &p2) {

        int nparts = Machine::config.num_of_partitions(p1.dims(), p1.bytes());
        auto sub_p1 = create_partitions(const_cast<Partition<T> &>(p1), nparts);
        auto sub_p2 = create_partitions(const_cast<Partition<T> &>(p2), nparts);

        // create a scheduler
        T sum = 0;
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> s(sub_p1, sub_p2);
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto &&[idx, d_p1, d_p2] = std::move(work.value());
                sum += d_p1.dot(d_p2);
            }
        }
        return sum;
    }

    template <typename T>
    T dot(const DArray<T> &a, const DArray<T> &b) {
        if (a.size() != b.size())
            throw std::runtime_error("dot: arrays must be the same size");

        int n_dev = Machine::config.num_of_gpus();
        T sum = 0;
        auto p1 = create_partitions(const_cast<DArray<T> &>(a), n_dev);
        auto p2 = create_partitions(const_cast<DArray<T> &>(b), n_dev);
        for (int i = 0; i < n_dev; ++i) { sum += dot(p1[i], p2[i]); }
        return sum;
    }

    // p1 = alpha * p1 + p2
    template <typename T>
    void axpy(Partition<T> &p1, T alpha, const Partition<T> &p2) {

        // create sub-partitions
        int n_parts = Machine::config.num_of_partitions(p1.dims(), p1.bytes());
        auto sub_p1 = create_partitions(p1, n_parts);
        auto sub_p2 = create_partitions(const_cast<Partition<T> &>(p2), n_parts);
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> s(sub_p1, sub_p2);
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto &&[idx, d_p1, d_p2] = std::move(work.value());
                gpu::axpy(d_p1.data(), alpha, d_p2.data(), d_p1.size());
                d_p1.copy_to(sub_p1[idx]);
            }
        }
    }

    // returns alpha * a + b (out-of-place, so it can bind to a temporary
    // returned by A(x) and be assigned to a fresh DArray, as cgsolver does)
    template <typename T>
    DArray<T> axpy(DArray<T> a, T alpha, const DArray<T> &b) {
        if (a.size() != b.size())
            throw std::runtime_error("axpy: arrays must be the same size");

        int n_dev = Machine::config.num_of_gpus();
        auto p1 = create_partitions(a, n_dev);
        auto p2 = create_partitions(const_cast<DArray<T> &>(b), n_dev);
        for (int i = 0; i < n_dev; ++i) { axpy(p1[i], alpha, p2[i]); }
        Machine::config.barrier();
        return a;
    }

    template <typename T>
    void xpay(Partition<T> &p1, T alpha, const Partition<T> &p2) {
        // create sub-partitions
        int n_parts = Machine::config.num_of_partitions(p1.dims(), p1.bytes());
        auto sub_p1 = create_partitions(p1, n_parts);
        auto sub_p2 = create_partitions(const_cast<Partition<T> &>(p2), n_parts);
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> s(sub_p1, sub_p2);
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto &&[idx, d_p1, d_p2] = std::move(work.value());
                gpu::xpay(d_p1.data(), alpha, d_p2.data(), d_p1.size());
                d_p1.copy_to(sub_p1[idx]);
            }
        }
    }

    // implements x += alpha * y
    template <typename T>
    void xpay(DArray<T> &a, T alpha, const DArray<T> &b) {
        if (a.size() != b.size())
            throw std::runtime_error("xpay: arrays must be the same size");

        int n_dev = Machine::config.num_of_gpus();
        auto p1 = create_partitions(a, n_dev);
        auto p2 = create_partitions(const_cast<DArray<T> &>(b), n_dev);
        for (int i = 0; i < n_dev; ++i) { xpay(p1[i], alpha, p2[i]); }
        Machine::config.barrier();
    }

} // namespace tomocam::array
