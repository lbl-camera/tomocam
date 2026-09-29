#include <cmath>
#include <functional>
#include <iostream>

#include "dist_array.h"
#include "optimize.h"
#include "test_utils.h"

// nagopt has no dedicated test elsewhere. Uses a small synthetic diagonal
// quadratic (self-adjoint, closed-form gradient/Lipschitz constant) to
// isolate the FISTA loop and backtracking line search from Radon/Toeplitz
// correctness.
int main() {
    // nslices must be >= Machine::config.num_of_gpus() -- array::dot's
    // partitioning doesn't clamp device count to nslices, so a smaller
    // value here can hit a divide-by-zero on multi-GPU machines.
    tomocam::dim3_t dims(16, 17, 17);

    auto scale = random_uniform<float>(dims, 0.5f, 2.f, 6); // diagonal weights
    auto x_true = random_uniform<float>(dims, -1.f, 1.f, 7);
    auto b = scale * x_true; // A(x_true), A(x) = scale * x

    std::function<float(tomocam::DArray<float> &)> loss =
        [&](tomocam::DArray<float> &x) {
            auto r = scale * x - b;
            return 0.5f * r.norm();
        };
    std::function<tomocam::DArray<float>(tomocam::DArray<float> &)> gradient =
        [&](tomocam::DArray<float> &x) {
            auto r = scale * x - b;
            return scale * r;
        };

    tomocam::DArray<float> x0(dims);
    x0.init(0.f);

    float L = scale.max() * scale.max(); // Lipschitz constant of the gradient
    float step_size = 1.f / L;

    tomocam::ReconParams params;
    params.max_iters = 60; // nagopt logs one line/iter; keep this modest

    auto x_rec = tomocam::nagopt<float>(gradient, loss, x0, step_size, params);

    double err = rel_error(x_rec, x_true);
    double loss0 = loss(x0);
    auto x_rec_copy = x_rec;
    double lossN = loss(x_rec_copy);
    std::cout << "nagopt rel_err: " << err << "  loss0: " << loss0
               << "  lossN: " << lossN << std::endl;

    bool ok = (err < 1e-2) && (lossN < 0.01 * loss0);
    return report(ok);
}
