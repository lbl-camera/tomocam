#include <functional>
#include <iostream>
#include <random>

#include "dist_array.h"
#include "optimize.h"
#include "tomocam.h"

int main() {
    tomocam::dim3_t dims = {4, 17, 17};
    tomocam::DArray<float> clean(dims), noisy(dims);

    std::mt19937 rng(7);
    std::normal_distribution<float> n(0.f, 0.05f);
    for (uint64_t i = 0; i < clean.size(); ++i) {
        clean[i] = (i % 5 < 2) ? 1.0f : 0.0f; // synthetic piecewise-constant signal
        noisy[i] = clean[i] + n(rng);
    }

    // identity operator: split_bregman degenerates to a pure TV denoiser
    std::function<tomocam::DArray<float>(tomocam::DArray<float> &)> A =
        [](tomocam::DArray<float> &v) { return v; };

    // mu (ADMM penalty) only affects convergence speed, not the fixed point;
    // lambda (TV weight) controls the actual denoising strength -- tuned
    // small here since the fixed point's TV bias grows quickly with lambda
    // for this data (confirmed by sweeping lambda from 0 to 0.05: err_after
    // is best around lambda ~ 1e-3 to 5e-3, and gets worse than the noisy
    // baseline above lambda ~ 1e-2).
    tomocam::Params params;
    params.max_iters = 50; // CG inner-loop cap
    params.tol = 1e-8;
    params.xtol = 1e-8;
    params.mu = 1.0;
    params.lambda = 0.001;
    params.outer_max = 60;

    auto x0 = noisy;
    auto rec = tomocam::split_bregman<float>(A, noisy, x0, params);

    auto err_before = (noisy - clean).norm();
    auto err_after = (rec - clean).norm();
    std::cout << "err before: " << err_before << "  err after: " << err_after
              << std::endl;

    bool ok = err_after < err_before;
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
