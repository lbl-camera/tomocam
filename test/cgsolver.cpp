#include <cmath>
#include <functional>
#include <iostream>

#include "dist_array.h"
#include "dist_array_ops.h"
#include "optimize.h"
#include "test_utils.h"

// cgsolver has no dedicated test elsewhere. Mirrors how split_bregman
// actually uses it: A(x) = x - mu * laplacian(x), an SPD operator built
// from array::laplacian (finitdiff_ops.cpp), solved against a known x_true.
int main() {
    // nslices must be >= Machine::config.num_of_gpus() -- array::dot's
    // partitioning doesn't clamp device count to nslices, so a smaller
    // value here can hit a divide-by-zero on multi-GPU machines.
    tomocam::dim3_t dims(16, 33, 33);
    float mu = 0.1f;

    std::function<tomocam::DArray<float>(tomocam::DArray<float> &)> A =
        [mu](tomocam::DArray<float> &v) {
            auto lap = tomocam::array::laplacian(v);
            return v - mu * lap;
        };

    auto x_true = random_uniform<float>(dims, -1.f, 1.f, 5);
    auto b = A(x_true);

    tomocam::DArray<float> x0(dims);
    x0.init(0.f);

    tomocam::ReconParams params;
    params.inner_iters = 200; // cgsolver's own iteration cap
    params.tol = 1e-6;        // cgsolver checks ||r||^2 < tol^2

    auto x_rec = tomocam::cgsolver<float>(A, b, x0, params);

    double err = rel_error(x_rec, x_true);
    std::cout << "cgsolver rel_err: " << err << std::endl;

    bool ok = err < 1e-3;
    return report(ok);
}
