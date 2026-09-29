/* Regression test: FinufftPlanWrapper::make_plan (src/finufft_plan.h) used
 * to hardcode ntrans=1 regardless of how many slices were actually in the
 * DeviceArray passed to nufft2d1/nufft2d2 (src/nufft.h), so every slice
 * beyond the first in a multi-slice batch was silently left untouched
 * (confirmed: relative error of exactly 1.0 against the never-written
 * output buffer).
 *
 * This backprojects a 2-slice sinogram in one batched call, and backprojects
 * each of its 2 slices separately as its own 1-slice call. The two must
 * agree slice-by-slice.
 */

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "dev_array.h"
#include "dist_array.h"
#include "internals.h"
#include "nufft.h"
#include "partition.h"
#include "types.h"

using real_t = float;
using namespace tomocam;

namespace {

real_t rel_diff(const std::vector<real_t> &a, const std::vector<real_t> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    }
    return static_cast<real_t>(std::sqrt(num / std::max(den, 1e-30)));
}

} // namespace

int main() {
    int nproj = 180, ncols = 128;
    std::vector<real_t> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * static_cast<real_t>(M_PI) / nproj;
    nufft::Grid<real_t> grid(nproj, ncols, theta.data(), 0);

    dim3_t sino1_dims(1, nproj, ncols);
    dim3_t sino2_dims(2, nproj, ncols);

    DArray<real_t> sino0(sino1_dims), sino1(sino1_dims), sino2(sino2_dims);
    std::mt19937 gen0(111), gen1(222);
    std::uniform_real_distribution<real_t> dist(0.0f, 1.0f);
    for (uint64_t i = 0; i < sino0.size(); i++) sino0[i] = dist(gen0);
    for (uint64_t i = 0; i < sino1.size(); i++) sino1[i] = dist(gen1);
    for (uint64_t i = 0; i < sino0.size(); i++) {
        sino2[i] = sino0[i];               // slice 0
        sino2[sino0.size() + i] = sino1[i]; // slice 1
    }

    auto p_sino0 = create_partitions(sino0, 1)[0];
    auto p_sino1 = create_partitions(sino1, 1)[0];
    auto p_sino2 = create_partitions(sino2, 1)[0];

    DeviceArray<real_t> d_sino0(p_sino0);
    DeviceArray<real_t> d_sino1(p_sino1);
    DeviceArray<real_t> d_sino2(p_sino2);

    // separate 1-slice calls (ground truth)
    auto d_recn0 = backproject(d_sino0, grid, real_t(0), false);
    auto d_recn1 = backproject(d_sino1, grid, real_t(0), false);

    // one batched 2-slice call
    auto d_recn2 = backproject(d_sino2, grid, real_t(0), false);

    auto h_recn0 = d_recn0.copy_to_host();
    auto h_recn1 = d_recn1.copy_to_host();
    auto h_recn2 = d_recn2.copy_to_host();

    size_t slice_sz = h_recn0.size();
    std::vector<real_t> batch_slice0(h_recn2.begin(), h_recn2.begin() + slice_sz);
    std::vector<real_t> batch_slice1(h_recn2.begin() + slice_sz, h_recn2.end());

    real_t err_slice0 = rel_diff(h_recn0, batch_slice0);
    real_t err_slice1 = rel_diff(h_recn1, batch_slice1);

    // sanity: the two inputs are genuinely different, so a correct fix
    // can't pass by both slices accidentally computing the same thing
    real_t sep_diff = rel_diff(h_recn0, h_recn1);

    std::printf("separate-slice0 vs batched-slice0 relative error: %.3e\n",
        err_slice0);
    std::printf("separate-slice1 vs batched-slice1 relative error: %.3e\n",
        err_slice1);
    std::printf(
        "separate-slice0 vs separate-slice1 (sanity, should be large): %.3e\n",
        sep_diff);

    constexpr real_t tol = 1e-4f;
    if (sep_diff < 1e-2f) {
        std::fprintf(stderr,
            "test setup failure: slice0/slice1 inputs weren't distinct "
            "enough to be a meaningful check\n");
        return 1;
    }
    if (err_slice0 > tol || err_slice1 > tol) {
        std::fprintf(stderr,
            "FAIL: batched multi-slice backproject() disagrees with "
            "separate single-slice calls (ntrans not threaded through "
            "correctly in the NUFFT plan)\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
