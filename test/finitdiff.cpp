#include <array>
#include <cmath>
#include <iostream>
#include <random>

#include "dist_array.h"
#include "dist_array_ops.h"
#include "types.h"

int main() {
    tomocam::dim3_t dims = {8, 33, 37}; // non-power-of-2, exercises boundary paths
    tomocam::DArray<float> x(dims);
    tomocam::DArray<float> y0(dims), y1(dims), y2(dims);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (uint64_t i = 0; i < x.size(); ++i) x[i] = u(rng);
    for (uint64_t i = 0; i < y0.size(); ++i) {
        y0[i] = u(rng);
        y1[i] = u(rng);
        y2[i] = u(rng);
    }

    auto gx = tomocam::array::grad_u(x);
    double lhs = 0;
    for (uint64_t i = 0; i < x.size(); ++i) {
        lhs += static_cast<double>(gx[0][i]) * y0[i];
        lhs += static_cast<double>(gx[1][i]) * y1[i];
        lhs += static_cast<double>(gx[2][i]) * y2[i];
    }

    std::array<tomocam::DArray<float>, 3> y{y0, y1, y2};
    auto div_y = tomocam::array::divergence(y);
    double rhs = 0;
    for (uint64_t i = 0; i < x.size(); ++i) {
        rhs += -1.0 * static_cast<double>(x[i]) * div_y[i];
    }

    // grad_u (forward diff, last slice per axis forced to 0 by Neumann
    // clamping) and divergence (backward diff, with both the "here" term at
    // the last index and the "previous" term at the first index treated as
    // 0, not clamped) are constructed to be the EXACT matrix transpose of
    // one another (up to the sign flip divergence = -grad_u^T) -- no
    // boundary correction should be needed; residual should be at float
    // accumulation precision.
    double residual = lhs - rhs;
    double rel_err = std::abs(residual) / (std::abs(lhs) + 1e-8);
    std::cout << "adjoint check: lhs=" << lhs << " rhs=" << rhs
              << " residual_rel_err=" << rel_err << std::endl;

    // laplacian sanity check: laplacian(constant array) == 0 everywhere
    tomocam::DArray<float> c(dims);
    c.init(1.0f);
    auto lap_c = tomocam::array::laplacian(c);
    double lap_max = std::max(std::abs(lap_c.max()), std::abs(lap_c.min()));
    std::cout << "laplacian(const) max abs: " << lap_max << std::endl;

    bool ok = (rel_err < 1e-3) && (lap_max < 1e-5);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
