/* Benchmark: chunk-count overhead for gradient2 (src/gradient2.cpp).
 *
 * Same question as bench_chunking.cpp, but for a compute-bound,
 * Toeplitz/FFT-based kernel instead of a memory-bound elementwise one, and
 * with gradient2_'s heavier pipeline: a Scheduler producer thread for H2D
 * *and* a separate GPUToHost shipper thread for D2H (two spawned threads
 * per call, vs. one for axpy/dot/xpay).
 *
 * Three configurations, mirroring gradient2_'s exact body:
 *   raw      - no Scheduler, no shipper: one H2D copy, one psf.convolve()
 *              call, one D2H copy.
 *   sched-1  - goes through Scheduler + GPUToHost with exactly 1 chunk.
 *   sched-N  - same path swept over explicit chunk counts.
 *
 * Usage: bench_gradient2 [--slices N] [--ncols N] [--nproj N] [--reps N]
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "nufft.h"
#include "partition.h"
#include "scheduler.h"
#include "shipper.h"
#include "timer.h"
#include "toeplitz.h"
#include "tomocam.h"
#include "types.h"

using real_t = float;
using tomocam::DArray;
using tomocam::DeviceArray;
using tomocam::GPUToHost;
using tomocam::Partition;
using tomocam::PointSpreadFunction;
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

// mirrors gradient2_'s body exactly (src/gradient2.cpp:37-67), but with an
// explicit chunk count instead of one computed from free memory, and no
// Scheduler/GPUToHost -- a single H2D copy, one kernel, one D2H copy
double run_raw(const Partition<real_t> &f, const Partition<real_t> &sinoT,
    Partition<real_t> &df, const PointSpreadFunction<real_t> &psf) {
    double t0 = now_ms();
    DeviceArray<real_t> d_f(f);
    DeviceArray<real_t> d_sinoT(sinoT);
    auto d_g = psf.convolve(d_f) - d_sinoT;
    d_g.copy_to(df);
    cudaDeviceSynchronize();
    return now_ms() - t0;
}

double run_sched(const Partition<real_t> &f, const Partition<real_t> &sinoT,
    Partition<real_t> &df, const PointSpreadFunction<real_t> &psf, int nparts) {
    auto p1 = create_partitions(const_cast<Partition<real_t> &>(f), nparts);
    auto p2 = create_partitions(const_cast<Partition<real_t> &>(sinoT), nparts);
    auto p3 = create_partitions(df, nparts);

    double t0 = now_ms();
    {
        GPUToHost<Partition<real_t>, DeviceArray<real_t>> shipper;
        Scheduler<Partition<real_t>, DeviceArray<real_t>, DeviceArray<real_t>> s(
            p1, p2);
        while (s.has_work()) {
            auto work = s.get_work();
            if (!work.has_value()) continue; // producer hasn't pushed yet
            auto &&[idx, d_f, d_sinoT] = std::move(work.value());
            auto d_g = psf.convolve(d_f) - d_sinoT;
            shipper.push(p3[idx], std::move(d_g));
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
    int nslices = 32, ncols = 512, nproj = 360, reps = 5;
    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char *flag) {
            return std::strcmp(argv[i], flag) == 0 && i + 1 < argc;
        };
        if (arg("--slices")) nslices = std::atoi(argv[++i]);
        else if (arg("--ncols")) ncols = std::atoi(argv[++i]);
        else if (arg("--nproj")) nproj = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::fprintf(stdout,
                "usage: bench_gradient2 [--slices N] [--ncols N] [--nproj N] "
                "[--reps N]\n");
            return 0;
        }
    }

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::fprintf(stdout, "free GPU memory: %.2f GB\n", free_mem / 1e9);
    std::fprintf(stdout,
        "recon dims: %d x %d x %d (%.2f GB per array), nproj=%d\n", nslices,
        ncols, ncols, static_cast<double>(nslices) * ncols * ncols *
            sizeof(real_t) / 1e9,
        nproj);

    // build the non-uniform grid and PSF once, same as mbir2()/gradient2()
    // do outside the optimization loop
    std::vector<real_t> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * static_cast<real_t>(M_PI) / nproj;
    tomocam::nufft::Grid<real_t> grid(nproj, ncols, theta.data(), 0);
    std::vector<PointSpreadFunction<real_t>> psfs;
    psfs.emplace_back(grid);
    const PointSpreadFunction<real_t> &psf = psfs[0];

    dim3_t dims(nslices, ncols, ncols);
    DArray<real_t> f(dims), sinoT(dims);
    NPRandom rng;
    for (uint64_t i = 0; i < f.size(); i++) {
        f[i] = rng.rand<real_t>();
        sinoT[i] = rng.rand<real_t>();
    }

    auto p_f = create_partitions(f, 1)[0];
    auto p_sinoT = create_partitions(sinoT, 1)[0];

    // total bytes moved per call: H2D(f) + H2D(sinoT) + D2H(df) -- doesn't
    // count convolve()'s internal padded-FFT workspace traffic, same
    // convention as bench_chunking.cpp so results are comparable
    double bytes_per_call = 3.0 * f.size() * sizeof(real_t);

    std::fprintf(stdout, "\n--- gradient2: psf.convolve(f) - sinoT ---\n");
    std::fprintf(stdout, "%-10s %6s %10s %10s %10s\n", "config", "chunks",
        "mean_ms", "min_ms", "GB/s");

    DArray<real_t> df_raw(dims);
    auto p_df_raw = create_partitions(df_raw, 1)[0];
    Stats raw_stats = time_it(
        [&]() { return run_raw(p_f, p_sinoT, p_df_raw, psf); }, reps,
        bytes_per_call);
    print_row("raw", 1, raw_stats);

    DArray<real_t> df_sched1(dims);
    auto p_df_sched1 = create_partitions(df_sched1, 1)[0];
    Stats sched1_stats = time_it(
        [&]() { return run_sched(p_f, p_sinoT, p_df_sched1, psf, 1); }, reps,
        bytes_per_call);
    print_row("sched", 1, sched1_stats);

    std::vector<int> chunk_counts = {2, 4, 8, 16, 32};
    DArray<real_t> df_check(dims);
    for (int n : chunk_counts) {
        auto p_df_check = create_partitions(df_check, 1)[0];
        Stats s = time_it(
            [&]() { return run_sched(p_f, p_sinoT, p_df_check, psf, n); }, reps,
            bytes_per_call);
        print_row("sched", n, s);
    }

    real_t err = rel_error(df_raw, df_sched1);
    std::fprintf(stdout, "raw vs sched-1 relative error: %.3e\n", err);
    err = rel_error(df_raw, df_check);
    std::fprintf(stdout,
        "raw vs sched-%d relative error: %.3e (correctness check)\n",
        chunk_counts.back(), err);

    // production API: tomocam::gradient2(DArray&, DArray&, vector<PSF>&),
    // which uses Machine::config.num_of_partitions internally
    std::fprintf(stdout, "\n--- production API (tomocam::gradient2) ---\n");
    int actual_nparts =
        tomocam::Machine::config.num_of_partitions(p_f.dims(), p_f.bytes());
    std::fprintf(stdout, "num_of_partitions() returns: %d chunk(s)\n",
        actual_nparts);

    Stats prod_stats = time_it(
        [&]() {
            double t0 = now_ms();
            auto g = tomocam::gradient2<real_t>(f, sinoT, psfs);
            cudaDeviceSynchronize();
            (void)g;
            return now_ms() - t0;
        },
        reps, bytes_per_call);
    print_row("prod", actual_nparts, prod_stats);

    return 0;
}
