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

#include "bench_support.h"
#include "dev_array.h"
#include "dist_array.h"
#include "dist_array_ops.h"
#include "gpu/gpu_ops.cuh"
#include "machine.h"
#include "partition.h"
#include "scheduler.h"
#include "test_utils.h"

using namespace bench;
using real_t = float;
using tomocam::DArray;
using tomocam::DeviceArray;
using tomocam::Partition;
using tomocam::Scheduler;
using tomocam::create_partitions;
using tomocam::dim3_t;

namespace {

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

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // stream results as they land
    int nslices = 128, nrows = 2048, ncols = 2048, reps = 10;
    parse_int_flags(argc, argv,
        {{"--slices", &nslices}, {"--rows", &nrows}, {"--cols", &ncols},
            {"--reps", &reps}},
        "usage: bench_chunking [--slices N] [--rows N] [--cols N] "
        "[--reps N]");

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

    real_t err = bench::rel_error(a_raw, a_sched1);
    std::fprintf(stdout, "raw vs sched-1 relative error: %.3e\n", err);
    err = bench::rel_error(a_raw, a_check);
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
