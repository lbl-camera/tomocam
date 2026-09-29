/* Benchmark: chunk-count overhead for function_value (src/error.cpp).
 *
 * The non-Toeplitz sibling of function_value2 -- uses a direct forward NUFFT
 * projection (project()) from image space to sinogram space instead of the
 * Toeplitz-based psf.convolve() shortcut, so recon (image space, ncols x
 * ncols) and sino (sinogram space, nproj x ncols) have different per-slice
 * shapes. Not currently in nagopt's call graph (only caller is the
 * deprecated src/mbir.cpp), added because an earlier benchmark found it
 * faster than function_value2 -- this sweep checks whether that holds once
 * chunk count is controlled for.
 *
 * Three configurations, mirroring funcval's exact body:
 *   raw      - no Scheduler: one H2D copy pair, one funcval body, done.
 *   sched-1  - goes through Scheduler with exactly 1 chunk.
 *   sched-N  - same path swept over explicit chunk counts.
 *
 * Usage: bench_function_value [--slices N] [--ncols N] [--nproj N] [--reps N]
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
#include "internals.h"
#include "machine.h"
#include "nufft.h"
#include "partition.h"
#include "scheduler.h"
#include "test_utils.h"
#include "tomocam.h"
#include "types.h"

using namespace bench;
using real_t = float;
using tomocam::DArray;
using tomocam::DeviceArray;
using tomocam::Partition;
using tomocam::Scheduler;
using tomocam::create_partitions;
using tomocam::dim3_t;

namespace {

// mirrors funcval's body exactly (src/error.cpp:36-63), but with an
// explicit chunk count instead of one computed from free memory
std::pair<double, real_t> run_raw(const Partition<real_t> &recon,
    const Partition<real_t> &sino, const tomocam::nufft::Grid<real_t> &grid) {
    double t0 = now_ms();
    DeviceArray<real_t> d_recon(recon);
    DeviceArray<real_t> d_sino(sino);
    auto t1 = tomocam::project(d_recon, grid);
    auto t2 = t1 - d_sino;
    real_t sum = t2.dot(t2);
    cudaDeviceSynchronize();
    return {now_ms() - t0, sum};
}

std::pair<double, real_t> run_sched(const Partition<real_t> &recon,
    const Partition<real_t> &sino, const tomocam::nufft::Grid<real_t> &grid,
    int nparts) {
    auto p1 = create_partitions(const_cast<Partition<real_t> &>(recon), nparts);
    auto p2 = create_partitions(const_cast<Partition<real_t> &>(sino), nparts);

    double t0 = now_ms();
    Scheduler<Partition<real_t>, DeviceArray<real_t>, DeviceArray<real_t>> s(
        p1, p2);
    real_t sum = 0;
    while (s.has_work()) {
        auto work = s.get_work();
        if (!work.has_value()) continue; // producer hasn't pushed yet
        auto &&[idx, d_recon, d_sino] = std::move(work.value());
        auto t1 = tomocam::project(d_recon, grid);
        auto t2 = t1 - d_sino;
        sum += t2.dot(t2);
    }
    cudaDeviceSynchronize();
    return {now_ms() - t0, sum};
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // stream results as they land
    int nslices = 32, ncols = 512, nproj = 360, reps = 5;
    parse_int_flags(argc, argv,
        {{"--slices", &nslices}, {"--ncols", &ncols}, {"--nproj", &nproj},
            {"--reps", &reps}},
        "usage: bench_function_value [--slices N] [--ncols N] "
        "[--nproj N] [--reps N]");

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::fprintf(stdout, "free GPU memory: %.2f GB\n", free_mem / 1e9);
    std::fprintf(stdout,
        "recon dims: %d x %d x %d, sino dims: %d x %d x %d (%.2f GB "
        "recon + %.2f GB sino), nproj=%d\n",
        nslices, ncols, ncols, nslices, nproj, ncols,
        static_cast<double>(nslices) * ncols * ncols * sizeof(real_t) / 1e9,
        static_cast<double>(nslices) * nproj * ncols * sizeof(real_t) / 1e9,
        nproj);

    // build the non-uniform grid once, same as mbir.cpp does outside the
    // optimization loop
    std::vector<real_t> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * static_cast<real_t>(M_PI) / nproj;
    tomocam::nufft::Grid<real_t> grid(nproj, ncols, theta.data(), 0);

    dim3_t recon_dims(nslices, ncols, ncols);
    dim3_t sino_dims(nslices, nproj, ncols);
    DArray<real_t> recon(recon_dims), sino(sino_dims);
    NPRandom rng;
    for (uint64_t i = 0; i < recon.size(); i++) recon[i] = rng.rand<real_t>();
    for (uint64_t i = 0; i < sino.size(); i++) sino[i] = rng.rand<real_t>();

    auto p_recon = create_partitions(recon, 1)[0];
    auto p_sino = create_partitions(sino, 1)[0];

    // total bytes moved per call: H2D(recon) + H2D(sino); no D2H -- the
    // output is a reduced scalar
    double bytes_per_call =
        static_cast<double>(recon.size() + sino.size()) * sizeof(real_t);

    std::fprintf(stdout,
        "\n--- function_value: sum((project(recon) - sino)^2) ---\n");
    std::fprintf(stdout, "%-10s %6s %10s %10s %10s\n", "config", "chunks",
        "mean_ms", "min_ms", "GB/s");

    real_t raw_val = 0, sched1_val = 0, sched_check_val = 0;
    Stats raw_stats = time_it(
        [&]() { return run_raw(p_recon, p_sino, grid); }, reps, bytes_per_call,
        &raw_val);
    print_row("raw", 1, raw_stats);

    Stats sched1_stats = time_it(
        [&]() { return run_sched(p_recon, p_sino, grid, 1); }, reps,
        bytes_per_call, &sched1_val);
    print_row("sched", 1, sched1_stats);

    std::vector<int> chunk_counts = {2, 4, 8, 16, 32};
    for (int n : chunk_counts) {
        Stats s = time_it(
            [&]() { return run_sched(p_recon, p_sino, grid, n); }, reps,
            bytes_per_call, &sched_check_val);
        print_row("sched", n, s);
        real_t rel = rel_error(raw_val, sched_check_val);
        std::fprintf(stdout, "  (relative error vs raw at %d chunks: %.3e)\n",
            n, rel);
    }

    real_t rel1 = rel_error(raw_val, sched1_val);
    std::fprintf(stdout, "raw vs sched-1 relative error: %.3e\n", rel1);
    real_t relN = rel_error(raw_val, sched_check_val);
    std::fprintf(stdout,
        "raw vs sched-%d relative error: %.3e (correctness check)\n",
        chunk_counts.back(), relN);

    // production API: tomocam::function_value(DArray&, DArray&,
    // vector<Grid>&), which uses Machine::config.num_of_partitions
    // internally
    std::fprintf(stdout,
        "\n--- production API (tomocam::function_value) ---\n");
    int actual_nparts = tomocam::Machine::config.num_of_partitions(
        p_recon.dims(), p_recon.bytes());
    std::fprintf(stdout, "num_of_partitions() returns: %d chunk(s)\n",
        actual_nparts);

    std::vector<tomocam::nufft::Grid<real_t>> grids;
    grids.push_back(std::move(grid)); // grid isn't needed again after this

    real_t prod_val = 0;
    Stats prod_stats = time_it(
        [&]() {
            double t0 = now_ms();
            real_t v = tomocam::function_value<real_t>(recon, sino, grids);
            cudaDeviceSynchronize();
            return std::make_pair(now_ms() - t0, v);
        },
        reps, bytes_per_call, &prod_val);
    print_row("prod", actual_nparts, prod_stats);

    return 0;
}
