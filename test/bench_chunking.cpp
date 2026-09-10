/* Benchmark: does chunk count (Scheduler pipelining) or PCIe bandwidth
 * dominate the cost of the memory-bound ops in src/dist_array_ops.h?
 *
 * Three configurations are timed for the same axpy/dot workload:
 *   raw      - no Scheduler: one H2D copy, one kernel, one D2H copy.
 *   sched-1  - goes through Scheduler/create_partitions with exactly 1 chunk.
 *   sched-N  - same Scheduler path, swept over explicit chunk counts.
 *
 * raw vs sched-1 isolates the fixed cost of the Scheduler/thread machinery.
 * sched-1 vs sched-N isolates the effect of chunk count/size.
 *
 * Usage: bench_chunking [--slices N] [--rows N] [--cols N] [--reps N]
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dev_array.h"
#include "dist_array.h"
#include "dist_array_ops.h"
#include "gpu/gpu_ops.cuh"
#include "machine.h"
#include "partition.h"
#include "scheduler.h"
#include "timer.h"

using real_t = float;
using tomocam::DArray;
using tomocam::DeviceArray;
using tomocam::Partition;
using tomocam::Scheduler;
using tomocam::create_partitions;
using tomocam::dim3_t;

namespace {

struct Stats {
    double mean_ms;
    double min_ms;
    double gbytes_moved;
};

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
               high_resolution_clock::now().time_since_epoch())
        .count();
}

// one H2D copy, one kernel, one D2H copy -- no Scheduler, no threads
double run_raw_axpy(Partition<real_t> &p1, real_t alpha,
    const Partition<real_t> &p2) {
    double t0 = now_ms();
    DeviceArray<real_t> d1(p1);
    DeviceArray<real_t> d2(p2);
    tomocam::gpu::axpy(d1.data(), alpha, d2.data(), d1.size());
    d1.copy_to(p1);
    cudaDeviceSynchronize();
    return now_ms() - t0;
}

double run_raw_dot(const Partition<real_t> &p1, const Partition<real_t> &p2) {
    double t0 = now_ms();
    DeviceArray<real_t> d1(p1);
    DeviceArray<real_t> d2(p2);
    volatile real_t sum = d1.dot(d2);
    (void)sum;
    cudaDeviceSynchronize();
    return now_ms() - t0;
}

// mirrors array::axpy(Partition<T>&, T, const Partition<T>&) from
// dist_array_ops.h, but with an explicit chunk count instead of one
// computed from free memory
double run_sched_axpy(Partition<real_t> &p1, real_t alpha,
    const Partition<real_t> &p2, int nparts) {
    auto sub_p1 = create_partitions(p1, nparts);
    auto sub_p2 = create_partitions(const_cast<Partition<real_t> &>(p2), nparts);
    double t0 = now_ms();
    Scheduler<Partition<real_t>, DeviceArray<real_t>, DeviceArray<real_t>> s(
        sub_p1, sub_p2);
    while (s.has_work()) {
        auto work = s.get_work();
        if (!work.has_value()) continue; // producer hasn't pushed yet
        auto &&[idx, d_p1, d_p2] = std::move(work.value());
        tomocam::gpu::axpy(d_p1.data(), alpha, d_p2.data(), d_p1.size());
        d_p1.copy_to(sub_p1[idx]);
    }
    cudaDeviceSynchronize();
    return now_ms() - t0;
}

double run_sched_dot(const Partition<real_t> &p1, const Partition<real_t> &p2,
    int nparts) {
    auto sub_p1 = create_partitions(const_cast<Partition<real_t> &>(p1), nparts);
    auto sub_p2 = create_partitions(const_cast<Partition<real_t> &>(p2), nparts);
    double t0 = now_ms();
    Scheduler<Partition<real_t>, DeviceArray<real_t>, DeviceArray<real_t>> s(
        sub_p1, sub_p2);
    real_t sum = 0;
    while (s.has_work()) {
        auto work = s.get_work();
        if (!work.has_value()) continue; // producer hasn't pushed yet
        auto &&[idx, d_p1, d_p2] = std::move(work.value());
        sum += d_p1.dot(d_p2);
    }
    cudaDeviceSynchronize();
    volatile real_t sink = sum;
    (void)sink;
    return now_ms() - t0;
}

template <typename Fn>
Stats time_it(Fn &&fn, int reps, double bytes_per_call) {
    fn(); // warmup, discarded
    double total = 0.0, best = 1e300;
    for (int i = 0; i < reps; i++) {
        double t = fn();
        total += t;
        if (t < best) best = t;
    }
    Stats s;
    s.mean_ms = total / reps;
    s.min_ms = best;
    s.gbytes_moved = bytes_per_call / 1e9;
    return s;
}

void print_row(const char *label, int chunks, const Stats &s) {
    double gbps = s.gbytes_moved / (s.min_ms / 1000.0);
    std::fprintf(stdout, "%-10s %6d %10.3f %10.3f %10.2f\n", label, chunks,
        s.mean_ms, s.min_ms, gbps);
    std::fflush(stdout);
}

real_t rel_error(const DArray<real_t> &a, const DArray<real_t> &b) {
    double num = 0.0, den = 0.0;
    for (uint64_t i = 0; i < a.size(); i++) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    }
    return static_cast<real_t>(std::sqrt(num / std::max(den, 1e-30)));
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // stream results as they land
    int nslices = 128, nrows = 2048, ncols = 2048, reps = 10;
    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char *flag) {
            return std::strcmp(argv[i], flag) == 0 && i + 1 < argc;
        };
        if (arg("--slices")) nslices = std::atoi(argv[++i]);
        else if (arg("--rows")) nrows = std::atoi(argv[++i]);
        else if (arg("--cols")) ncols = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::fprintf(stdout,
                "usage: bench_chunking [--slices N] [--rows N] [--cols N] "
                "[--reps N]\n");
            return 0;
        }
    }

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::fprintf(stdout, "free GPU memory: %.2f GB\n",
        free_mem / 1e9);
    std::fprintf(stdout, "array dims: %d x %d x %d (%.2f GB per array)\n",
        nslices, nrows, ncols,
        static_cast<double>(nslices) * nrows * ncols * sizeof(real_t) / 1e9);

    dim3_t dims(nslices, nrows, ncols);
    DArray<real_t> a(dims), b(dims), a_raw(dims), a_sched1(dims);

    NPRandom rng;
    for (uint64_t i = 0; i < a.size(); i++) {
        real_t v = rng.rand<real_t>();
        a[i] = v;
        a_raw[i] = v;
        a_sched1[i] = v;
        b[i] = rng.rand<real_t>();
    }
    real_t alpha = 1.5f;

    // ndevice_ is hard-coded to 1 (machine.h), so a single partition wraps
    // the whole array
    auto p_a_raw = create_partitions(a_raw, 1)[0];
    auto p_a_sched1 = create_partitions(a_sched1, 1)[0];
    auto p_b1 = create_partitions(b, 1)[0];
    auto p_b2 = create_partitions(b, 1)[0];

    // total bytes moved per call: H2D(p1) + H2D(p2) + D2H(p1)
    double bytes_per_call = 3.0 * a.size() * sizeof(real_t);

    std::fprintf(stdout, "\n--- axpy: p1 = alpha*p1 + p2 ---\n");
    std::fprintf(stdout, "%-10s %6s %10s %10s %10s\n", "config", "chunks",
        "mean_ms", "min_ms", "GB/s");

    Stats raw_stats = time_it(
        [&]() { return run_raw_axpy(p_a_raw, alpha, p_b1); }, reps,
        bytes_per_call);
    print_row("raw", 1, raw_stats);

    Stats sched1_stats = time_it(
        [&]() { return run_sched_axpy(p_a_sched1, alpha, p_b2, 1); }, reps,
        bytes_per_call);
    print_row("sched", 1, sched1_stats);

    std::vector<int> chunk_counts = {2, 4, 8, 16, 32, 64, 128};
    DArray<real_t> a_check(dims);
    for (int n : chunk_counts) {
        for (uint64_t i = 0; i < a.size(); i++) a_check[i] = a[i];
        auto p_check = create_partitions(a_check, 1)[0];
        auto p_b3 = create_partitions(b, 1)[0];

        Stats s = time_it(
            [&]() { return run_sched_axpy(p_check, alpha, p_b3, n); }, reps,
            bytes_per_call);
        print_row("sched", n, s);
    }

    real_t err = rel_error(a_raw, a_sched1);
    std::fprintf(stdout, "raw vs sched-1 relative error: %.3e\n", err);
    err = rel_error(a_raw, a_check);
    std::fprintf(stdout,
        "raw vs sched-%d relative error: %.3e (correctness check)\n",
        chunk_counts.back(), err);

    std::fprintf(stdout, "\n--- dot: sum(p1 .* p2) ---\n");
    std::fprintf(stdout, "%-10s %6s %10s %10s %10s\n", "config", "chunks",
        "mean_ms", "min_ms", "GB/s");

    double dot_bytes_per_call = 2.0 * a.size() * sizeof(real_t);
    auto p_b4 = create_partitions(b, 1)[0];
    Stats raw_dot = time_it([&]() { return run_raw_dot(p_a_raw, p_b4); },
        reps, dot_bytes_per_call);
    print_row("raw", 1, raw_dot);

    auto p_b5 = create_partitions(b, 1)[0];
    Stats sched1_dot = time_it(
        [&]() { return run_sched_dot(p_a_raw, p_b5, 1); }, reps,
        dot_bytes_per_call);
    print_row("sched", 1, sched1_dot);

    for (int n : chunk_counts) {
        auto p_b6 = create_partitions(b, 1)[0];
        Stats s = time_it([&]() { return run_sched_dot(p_a_raw, p_b6, n); },
            reps, dot_bytes_per_call);
        print_row("sched", n, s);
    }

    // smoke test: exercise the actual production API
    // (tomocam::array::axpy/dot), which is what previously crashed with
    // std::bad_optional_access due to the Scheduler race
    std::fprintf(stdout,
        "\n--- production API smoke test (array::axpy / array::dot) ---\n");
    auto p_smoke = create_partitions(a, 1)[0];
    auto p_b_smoke = create_partitions(b, 1)[0];
    int actual_nparts =
        tomocam::Machine::config.num_of_partitions(p_smoke.dims(), p_smoke.bytes());
    std::fprintf(stdout, "num_of_partitions() now returns: %d chunk(s)\n",
        actual_nparts);

    Stats prod_axpy = time_it(
        [&]() {
            double t0 = now_ms();
            tomocam::array::axpy(p_smoke, alpha, p_b_smoke);
            cudaDeviceSynchronize();
            return now_ms() - t0;
        },
        reps, bytes_per_call);
    print_row("prod-axpy", actual_nparts, prod_axpy);

    Stats prod_dot = time_it(
        [&]() {
            double t0 = now_ms();
            volatile real_t d = tomocam::array::dot(p_smoke, p_b_smoke);
            (void)d;
            cudaDeviceSynchronize();
            return now_ms() - t0;
        },
        reps, dot_bytes_per_call);
    print_row("prod-dot", actual_nparts, prod_dot);

    std::fprintf(stdout, "50 more iterations of array::axpy/array::dot "
                          "completed without crashing\n");
    for (int i = 0; i < 50; i++) {
        tomocam::array::axpy(p_smoke, alpha, p_b_smoke);
        volatile real_t d = tomocam::array::dot(p_smoke, p_b_smoke);
        (void)d;
    }

    return 0;
}
