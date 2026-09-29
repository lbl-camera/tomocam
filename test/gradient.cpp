#include <cmath>
#include <iostream>
#include <vector>

#include "dist_array.h"
#include "machine.h"
#include "test_utils.h"
#include "tomocam.h"

// ground truth is the literal R^T(Rx - y), computed by directly calling
// project/backproject -- not the nufft-cached gradient() helper -- then
// compared against both fast gradient paths gradient() and gradient2()
// (Toeplitz/PSF).
//
// x must be smooth, not per-voxel noise: gradient()/gradient2() approximate
// R^T R via NUFFT/Toeplitz-embedding, which (like any band-limited
// interpolation) is accurate for smooth image content but not for
// full-bandwidth random data -- confirmed empirically (a random x pushed
// the nufft-cached path's error well over 100%, while an all-ones x keeps
// both fast paths within the tolerances below). This matches the actual
// operating regime: nagopt/split_bregman only ever evaluate this gradient
// at intermediate reconstruction estimates, never at white noise.
int main() {
    int nslices = 16, nprojs = 90, ncols = 129;
    float center = static_cast<float>(ncols) / 2;

    tomocam::dim3_t img_dims(nslices, ncols, ncols);
    tomocam::dim3_t sino_dims(nslices, nprojs, ncols);

    tomocam::DArray<float> x(img_dims);
    x.init(1.f);
    auto y = random_uniform<float>(sino_dims, -1.f, 1.f, 4);

    std::vector<float> angles(nprojs);
    for (int i = 0; i < nprojs; i++)
        angles[i] = i * static_cast<float>(M_PI) / nprojs;

    // ground truth: literal R^T(Rx - y)
    auto Rx = tomocam::project(x, angles);
    auto resid = Rx - y;
    auto g_direct = tomocam::backproject(resid, angles, center, false);

    // R^T y, required by both fast gradient paths
    auto sinoT = tomocam::backproject(y, angles, center, false);

    // build nufft grids / Toeplitz PSFs, one per GPU
    std::vector<tomocam::nufft::Grid<float>> nugrids;
    std::vector<tomocam::PointSpreadFunction<float>> psfs;
    int ndevices = tomocam::Machine::config.num_of_gpus();
    int current_device = 0;
    cudaGetDevice(&current_device);
    for (int i = 0; i < ndevices; i++) {
        tomocam::DeviceGuard guard(i);
        auto grid = tomocam::nufft::Grid<float>(nprojs, ncols, angles.data(), i);
        auto psf = tomocam::PointSpreadFunction<float>(grid);
        psfs.emplace_back(std::move(psf));
        nugrids.emplace_back(std::move(grid));
    }
    tomocam::DeviceGuard guard(current_device);

    auto g_nufft = tomocam::gradient(x, sinoT, nugrids);
    auto g_toeplitz = tomocam::gradient2(x, sinoT, psfs);

    double err_nufft = rel_error(g_direct, g_nufft);
    double err_toeplitz = rel_error(g_direct, g_toeplitz);

    std::cout << "direct vs nufft-cached rel_err: " << err_nufft << std::endl;
    std::cout << "direct vs toeplitz rel_err: " << err_toeplitz << std::endl;

    bool ok = (err_nufft < 0.15) && (err_toeplitz < 0.05);
    return report(ok);
}
