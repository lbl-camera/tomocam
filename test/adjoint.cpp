#include <cmath>
#include <iostream>
#include <vector>

#include "dist_array.h"
#include "dist_array_ops.h"
#include "test_utils.h"
#include "tomocam.h"

// verifies that backproject() is the true adjoint of project(), i.e.
//   <R x, y> == <x, R^T y>
// for a random image x and a random sinogram y -- the property every other
// op (gradient, cgsolver, split_bregman) implicitly relies on.
int main() {
    // nslices must be >= Machine::config.num_of_gpus() -- array::dot's
    // partitioning doesn't clamp device count to nslices, so a smaller
    // value here can hit a divide-by-zero on multi-GPU machines.
    // ncols must be even: backproject's internal offset is
    // ncols/2 (integer) - center, so an odd ncols with center=ncols/2.0f
    // leaves a spurious half-pixel offset that breaks exact adjointness.
    int nslices = 16, ncols = 64, nproj = 90;
    float center = static_cast<float>(ncols) / 2;

    tomocam::dim3_t img_dims(nslices, ncols, ncols);
    tomocam::dim3_t sino_dims(nslices, nproj, ncols);

    auto x = random_uniform<float>(img_dims, -1.f, 1.f, 1);
    auto y = random_uniform<float>(sino_dims, -1.f, 1.f, 2);

    std::vector<float> angles(nproj);
    for (int i = 0; i < nproj; i++) angles[i] = i * static_cast<float>(M_PI) / nproj;

    auto Rx = tomocam::project(x, angles);
    auto Rty = tomocam::backproject(y, angles, center, false);

    double lhs = tomocam::array::dot(Rx, y);
    double rhs = tomocam::array::dot(x, Rty);

    double rel_err = std::abs(lhs - rhs) / (std::abs(lhs) + 1e-8);
    std::cout << "<Rx,y>=" << lhs << " <x,R^Ty>=" << rhs
               << " rel_err=" << rel_err << std::endl;

    bool ok = rel_err < 1e-3;
    return report(ok);
}
