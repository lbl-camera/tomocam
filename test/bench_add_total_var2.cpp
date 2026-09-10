/* Benchmark: chunk-count overhead for add_total_var2 (src/tv_update.cpp).
 *
 * Unlike axpy/dot/gradient2/function_value2, this kernel partitions its
 * *input* (sol) with a halo of 1 slice between chunks (create_partitions(sol,
 * nparts, 1)) since the TV gradient is a local stencil that needs
 * neighboring-slice access -- the *output* (grad) is partitioned without a
 * halo. That means smaller chunks move strictly more total bytes than
 * larger ones (redundant halo slices at every chunk boundary), an extra
 * cost axis beyond the fixed per-chunk overhead the other benchmarks
 * measured. `grad` is mutated in place (grad += tv_gradient(sol)), so
 * `sol` is read-only across repetitions but `grad` accumulates -- matching
 * bench_chunking.cpp's pattern, separate grad_raw/grad_sched1/grad_check
 * buffers get the exact same sequence of calls so their final states are
 * directly comparable.
 *
 * Usage: bench_add_total_var2 [--slices N] [--ncols N] [--reps N]
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dev_array.h"
#include "dist_array.h"
#include "gpu/totalvar.cuh"
#include "machine.h"
#include "partition.h"
#include "scheduler.h"
#include "shipper.h"
#include "timer.h"
#include "tomocam.h"
#include "types.h"

using real_t = float;
using tomocam::DArray;
using tomocam::DeviceArray;
using tomocam::GPUToHost;
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

// mirrors total_var2's body exactly (src/tv_update.cpp:42-72), but with an
// explicit chunk count instead of one computed from free memory; no halo
// needed since a single chunk covers the whole partition
double run_raw(const Partition<real_t> &sol, Partition<real_t> &grad, real_t sigma,
    real_t p) {
    double t0 = now_ms();
    DeviceArray<real_t> d_s(sol);
    DeviceArray<real_t> d_g(grad);
    tomocam::gpu::add_total_var2<real_t>(d_s, d_g, sigma, p);
    d_g.copy_to(grad);
    cudaDeviceSynchronize();
    return now_ms() - t0;
}

double run_sched(Partition<real_t> &sol, Partition<real_t> &grad, real_t sigma,
    real_t p, int nparts) {
    auto sub_sols = create_partitions(sol, nparts, 1); // halo=1 between chunks
    auto sub_grads = create_partitions(grad, nparts);  // no halo

    double t0 = now_ms();
    {
        GPUToHost<Partition<real_t>, DeviceArray<real_t>> shipper;
        Scheduler<Partition<real_t>, DeviceArray<real_t>, DeviceArray<real_t>> s(
            sub_sols, sub_grads);
        while (s.has_work()) {
            auto work = s.get_work();
            if (!work.has_value()) continue; // producer hasn't pushed yet
            auto &&[idx, d_s, d_g] = std::move(work.value());
            tomocam::gpu::add_total_var2<real_t>(d_s, d_g, sigma, p);
            shipper.push(sub_grads[idx], std::move(d_g));
        }
    } // ~GPUToHost() blocks until all pending D2H copies land
    cudaDeviceSynchronize();
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
    int nslices = 32, ncols = 512, reps = 5;
    real_t sigma = 200.0f, p = 1.2f;
    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char *flag) {
            return std::strcmp(argv[i], flag) == 0 && i + 1 < argc;
        };
        if (arg("--slices")) nslices = std::atoi(argv[++i]);
        else if (arg("--ncols")) ncols = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::fprintf(stdout,
                "usage: bench_add_total_var2 [--slices N] [--ncols N] "
                "[--reps N]\n");
            return 0;
        }
    }

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::fprintf(stdout, "free GPU memory: %.2f GB\n", free_mem / 1e9);
    std::fprintf(stdout, "recon dims: %d x %d x %d (%.2f GB per array)\n",
        nslices, ncols, ncols,
        static_cast<double>(nslices) * ncols * ncols * sizeof(real_t) / 1e9);

    dim3_t dims(nslices, ncols, ncols);
    DArray<real_t> sol(dims), grad_raw(dims), grad_sched1(dims);
    NPRandom rng;
    for (uint64_t i = 0; i < sol.size(); i++) {
        real_t v = rng.rand<real_t>();
        sol[i] = v;
        grad_raw[i] = rng.rand<real_t>();
    }
    for (uint64_t i = 0; i < sol.size(); i++) grad_sched1[i] = grad_raw[i];

    auto p_sol = create_partitions(sol, 1)[0];
    auto p_grad_raw = create_partitions(grad_raw, 1)[0];
    auto p_grad_sched1 = create_partitions(grad_sched1, 1)[0];

    // nominal bytes moved per call: H2D(sol) + H2D(grad) + D2H(grad); at
    // higher chunk counts sol actually moves more than this due to halo
    // overlap -- see the note printed after the sweep
    double bytes_per_call = 3.0 * sol.size() * sizeof(real_t);

    std::fprintf(stdout, "\n--- add_total_var2: grad += tv_gradient(sol) ---\n");
    std::fprintf(stdout, "%-10s %6s %10s %10s %10s\n", "config", "chunks",
        "mean_ms", "min_ms", "GB/s");

    Stats raw_stats = time_it(
        [&]() { return run_raw(p_sol, p_grad_raw, sigma, p); }, reps,
        bytes_per_call);
    print_row("raw", 1, raw_stats);

    Stats sched1_stats = time_it(
        [&]() { return run_sched(p_sol, p_grad_sched1, sigma, p, 1); }, reps,
        bytes_per_call);
    print_row("sched", 1, sched1_stats);

    std::vector<int> chunk_counts = {2, 4, 8, 16, 32};
    DArray<real_t> grad_check(dims);
    for (int n : chunk_counts) {
        for (uint64_t i = 0; i < sol.size(); i++) grad_check[i] = grad_raw[i];
        auto p_grad_check = create_partitions(grad_check, 1)[0];
        Stats s = time_it(
            [&]() { return run_sched(p_sol, p_grad_check, sigma, p, n); }, reps,
            bytes_per_call);
        print_row("sched", n, s);
    }

    real_t err = rel_error(grad_raw, grad_sched1);
    std::fprintf(stdout, "raw vs sched-1 relative error: %.3e\n", err);
    err = rel_error(grad_raw, grad_check);
    std::fprintf(stdout,
        "raw vs sched-%d relative error: %.3e (correctness check)\n",
        chunk_counts.back(), err);

    // quantify the halo overhead directly: at the highest chunk count,
    // how many extra sol-slices (redundant halo) were actually moved
    // compared to the nominal, halo-free byte count?
    int n_hi = chunk_counts.back();
    auto sub_sols_hi = create_partitions(p_sol, n_hi, 1);
    uint64_t halo_slices = 0;
    for (auto &part : sub_sols_hi) halo_slices += part.nslices();
    uint64_t nominal_slices = static_cast<uint64_t>(nslices);
    std::fprintf(stdout,
        "\nhalo overhead at %d chunks: %lu sol-slices moved vs %lu nominal "
        "(+%.1f%%)\n",
        n_hi, halo_slices, nominal_slices,
        100.0 * (static_cast<double>(halo_slices) / nominal_slices - 1.0));

    // production API: tomocam::add_total_var2(DArray&, DArray&, sigma, p),
    // which uses Machine::config.num_of_partitions internally
    std::fprintf(stdout, "\n--- production API (tomocam::add_total_var2) ---\n");
    int actual_nparts =
        tomocam::Machine::config.num_of_partitions(p_sol.dims(), p_sol.bytes());
    std::fprintf(stdout, "num_of_partitions() returns: %d chunk(s)\n",
        actual_nparts);

    DArray<real_t> grad_prod(dims);
    for (uint64_t i = 0; i < sol.size(); i++) grad_prod[i] = grad_raw[i];
    Stats prod_stats = time_it(
        [&]() {
            double t0 = now_ms();
            tomocam::add_total_var2<real_t>(sol, grad_prod, sigma, p);
            cudaDeviceSynchronize();
            return now_ms() - t0;
        },
        reps, bytes_per_call);
    print_row("prod", actual_nparts, prod_stats);

    return 0;
}
