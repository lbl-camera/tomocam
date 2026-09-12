/* Benchmark: chunk-count overhead for function_value2 (src/error2.cpp).
 *
 * Same question as bench_gradient2.cpp: compute-bound, Toeplitz/FFT-based
 * kernel, but funcval2's pipeline is lighter than gradient2_'s -- output is
 * a reduced scalar (no D2H array copy, no GPUToHost shipper), same pattern
 * as array::dot in bench_chunking.cpp, just with a heavier per-chunk body
 * (one psf.convolve() plus two dot products instead of one).
 *
 * Three configurations, mirroring funcval2's exact body:
 *   raw      - no Scheduler: one H2D copy pair, one funcval2 body, done.
 *   sched-1  - goes through Scheduler with exactly 1 chunk.
 *   sched-N  - same path swept over explicit chunk counts.
 *
 * Usage: bench_function_value2 [--slices N] [--ncols N] [--nproj N] [--reps N]
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
#include "machine.h"
#include "nufft.h"
#include "partition.h"
#include "scheduler.h"
#include "test_utils.h"
#include "toeplitz.h"
#include "tomocam.h"
#include "types.h"

using namespace bench;
using real_t = float;
using tomocam::DArray;
using tomocam::DeviceArray;
using tomocam::Partition;
using tomocam::PointSpreadFunction;
using tomocam::Scheduler;
using tomocam::create_partitions;
using tomocam::dim3_t;

namespace {

// mirrors funcval2's body exactly (src/error2.cpp:40-67), but with an
// explicit chunk count instead of one computed from free memory
std::pair<double, real_t> run_raw(const Partition<real_t> &recon,
    const Partition<real_t> &sinoT, const PointSpreadFunction<real_t> &psf) {
    double t0 = now_ms();
    DeviceArray<real_t> d_recon(recon);
    DeviceArray<real_t> d_sinoT(sinoT);
    auto t1 = psf.convolve(d_recon);
    auto t2 = d_recon.dot(t1);
    auto t3 = d_recon.dot(d_sinoT);
    real_t sum = t2 - 2 * t3;
    cudaDeviceSynchronize();
    return {now_ms() - t0, sum};
}

std::pair<double, real_t> run_sched(const Partition<real_t> &recon,
    const Partition<real_t> &sinoT, const PointSpreadFunction<real_t> &psf,
    int nparts) {
    auto p1 = create_partitions(const_cast<Partition<real_t> &>(recon), nparts);
    auto p2 = create_partitions(const_cast<Partition<real_t> &>(sinoT), nparts);

    double t0 = now_ms();
    Scheduler<Partition<real_t>, DeviceArray<real_t>, DeviceArray<real_t>> s(
        p1, p2);
    real_t sum = 0;
    while (s.has_work()) {
        auto work = s.get_work();
        if (!work.has_value()) continue; // producer hasn't pushed yet
        auto &&[idx, d_recon, d_sinoT] = std::move(work.value());
        auto t1 = psf.convolve(d_recon);
        auto t2 = d_recon.dot(t1);
        auto t3 = d_recon.dot(d_sinoT);
        sum += (t2 - 2 * t3);
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
        "usage: bench_function_value2 [--slices N] [--ncols N] "
        "[--nproj N] [--reps N]");

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::fprintf(stdout, "free GPU memory: %.2f GB\n", free_mem / 1e9);
    std::fprintf(stdout,
        "recon dims: %d x %d x %d (%.2f GB per array), nproj=%d\n", nslices,
        ncols, ncols, static_cast<double>(nslices) * ncols * ncols *
            sizeof(real_t) / 1e9,
        nproj);

    // build the non-uniform grid and PSF once, same as mbir2()/error2() do
    // outside the optimization loop
    std::vector<real_t> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * static_cast<real_t>(M_PI) / nproj;
    tomocam::nufft::Grid<real_t> grid(nproj, ncols, theta.data(), 0);
    std::vector<PointSpreadFunction<real_t>> psfs;
    psfs.emplace_back(grid);
    const PointSpreadFunction<real_t> &psf = psfs[0];

    dim3_t dims(nslices, ncols, ncols);
    DArray<real_t> recon(dims), sinoT(dims);
    NPRandom rng;
    for (uint64_t i = 0; i < recon.size(); i++) {
        recon[i] = rng.rand<real_t>();
        sinoT[i] = rng.rand<real_t>();
    }

    auto p_recon = create_partitions(recon, 1)[0];
    auto p_sinoT = create_partitions(sinoT, 1)[0];

    // total bytes moved per call: H2D(recon) + H2D(sinoT); no D2H -- the
    // output is a reduced scalar, same convention as bench_chunking.cpp's
    // dot benchmark
    double bytes_per_call = 2.0 * recon.size() * sizeof(real_t);

    std::fprintf(stdout,
        "\n--- function_value2: sum(psf.convolve(recon).dot(recon) - "
        "2*recon.dot(sinoT)) ---\n");
    std::fprintf(stdout, "%-10s %6s %10s %10s %10s\n", "config", "chunks",
        "mean_ms", "min_ms", "GB/s");

    real_t raw_val = 0, sched1_val = 0, sched_check_val = 0;
    Stats raw_stats = time_it(
        [&]() { return run_raw(p_recon, p_sinoT, psf); }, reps, bytes_per_call,
        &raw_val);
    print_row("raw", 1, raw_stats);

    Stats sched1_stats = time_it(
        [&]() { return run_sched(p_recon, p_sinoT, psf, 1); }, reps,
        bytes_per_call, &sched1_val);
    print_row("sched", 1, sched1_stats);

    std::vector<int> chunk_counts = {2, 4, 8, 16, 32};
    for (int n : chunk_counts) {
        Stats s = time_it(
            [&]() { return run_sched(p_recon, p_sinoT, psf, n); }, reps,
            bytes_per_call, &sched_check_val);
        print_row("sched", n, s);
    }

    real_t rel1 = rel_error(raw_val, sched1_val);
    std::fprintf(stdout, "raw vs sched-1 relative error: %.3e\n", rel1);
    real_t relN = rel_error(raw_val, sched_check_val);
    std::fprintf(stdout,
        "raw vs sched-%d relative error: %.3e (correctness check)\n",
        chunk_counts.back(), relN);

    // production API: tomocam::function_value2(DArray&, DArray&,
    // vector<PSF>&, sino_sq), which uses Machine::config.num_of_partitions
    // internally. sino_sq is just an additive constant here.
    std::fprintf(stdout,
        "\n--- production API (tomocam::function_value2) ---\n");
    int actual_nparts = tomocam::Machine::config.num_of_partitions(
        p_recon.dims(), p_recon.bytes());
    std::fprintf(stdout, "num_of_partitions() returns: %d chunk(s)\n",
        actual_nparts);

    real_t prod_val = 0;
    Stats prod_stats = time_it(
        [&]() {
            double t0 = now_ms();
            real_t v =
                tomocam::function_value2<real_t>(recon, sinoT, psfs, real_t(0));
            cudaDeviceSynchronize();
            return std::make_pair(now_ms() - t0, v);
        },
        reps, bytes_per_call, &prod_val);
    print_row("prod", actual_nparts, prod_stats);

    return 0;
}
