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

#include "bench_support.h"
#include "dev_array.h"
#include "dist_array.h"
#include "gpu/totalvar.cuh"
#include "machine.h"
#include "partition.h"
#include "scheduler.h"
#include "shipper.h"
#include "test_utils.h"
#include "tomocam.h"
#include "types.h"

using namespace bench;
using real_t = float;
using tomocam::DArray;
using tomocam::DeviceArray;
using tomocam::GPUToHost;
using tomocam::Partition;
using tomocam::Scheduler;
using tomocam::create_partitions;
using tomocam::dim3_t;

namespace {

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

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // stream results as they land
    int nslices = 32, ncols = 512, reps = 5;
    real_t sigma = 200.0f, p = 1.2f;
    parse_int_flags(argc, argv,
        {{"--slices", &nslices}, {"--ncols", &ncols}, {"--reps", &reps}},
        "usage: bench_add_total_var2 [--slices N] [--ncols N] "
        "[--reps N]");

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

    real_t err = bench::rel_error(grad_raw, grad_sched1);
    std::fprintf(stdout, "raw vs sched-1 relative error: %.3e\n", err);
    err = bench::rel_error(grad_raw, grad_check);
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
