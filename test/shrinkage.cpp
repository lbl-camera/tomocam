#include <array>
#include <cmath>
#include <iostream>
#include <random>

#include "dist_array.h"
#include "dist_array_ops.h"
#include "types.h"

// Reference CPU implementation of the isotropic TV shrinkage + Bregman
// update, matching the formula that used to live directly in
// tomocam::split_bregman (src/bregman.cpp) before it was ported to GPU.
template <typename T>
static void reference_shrinkage(const std::array<tomocam::DArray<T>, 3> &dx,
    std::array<tomocam::DArray<T>, 3> &bregman_b,
    std::array<tomocam::DArray<T>, 3> &d, T lambda_mu, T epsilon) {

    uint64_t n = dx[0].size();
    for (uint64_t i = 0; i < n; ++i) {
        T sum_sq = 0;
        for (int j = 0; j < 3; ++j) {
            T v = dx[j][i] + bregman_b[j][i];
            sum_sq += v * v;
        }
        T sk = std::sqrt(sum_sq);
        for (int j = 0; j < 3; ++j) {
            T v = (dx[j][i] + bregman_b[j][i]) / (sk + epsilon);
            T dij = std::max(T(0), sk - lambda_mu) * v;
            bregman_b[j][i] += dx[j][i] - dij;
            d[j][i] = dij;
        }
    }
}

int main() {
    tomocam::dim3_t dims = {8, 33, 37}; // non-power-of-2, matches finitdiff.cpp

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.f, 1.f);

    auto random_array = [&]() {
        tomocam::DArray<float> a(dims);
        for (uint64_t i = 0; i < a.size(); ++i) a[i] = u(rng);
        return a;
    };

    std::array<tomocam::DArray<float>, 3> dx{random_array(), random_array(),
        random_array()};

    // two independent copies of bregman_b/d: one path goes through the
    // reference CPU formula, the other through the new GPU kernel
    std::array<tomocam::DArray<float>, 3> b_ref{random_array(), random_array(),
        random_array()};
    std::array<tomocam::DArray<float>, 3> b_gpu{b_ref[0], b_ref[1], b_ref[2]};

    auto zeros = [&]() {
        tomocam::DArray<float> a(dims);
        a.init(0.f);
        return a;
    };
    std::array<tomocam::DArray<float>, 3> d_ref{zeros(), zeros(), zeros()};
    std::array<tomocam::DArray<float>, 3> d_gpu{zeros(), zeros(), zeros()};

    float lambda_mu = 0.37f;
    float epsilon = 1e-8f;

    reference_shrinkage<float>(dx, b_ref, d_ref, lambda_mu, epsilon);
    tomocam::array::shrinkage<float>(dx, b_gpu, d_gpu, lambda_mu, epsilon);

    double max_d_err = 0, max_b_err = 0;
    for (int j = 0; j < 3; ++j) {
        for (uint64_t i = 0; i < dx[0].size(); ++i) {
            max_d_err = std::max(max_d_err,
                static_cast<double>(std::abs(d_ref[j][i] - d_gpu[j][i])));
            max_b_err = std::max(max_b_err,
                static_cast<double>(std::abs(b_ref[j][i] - b_gpu[j][i])));
        }
    }

    std::cout << "max |d_ref - d_gpu|: " << max_d_err << std::endl;
    std::cout << "max |bregman_b_ref - bregman_b_gpu|: " << max_b_err
               << std::endl;

    bool ok = (max_d_err < 1e-5) && (max_b_err < 1e-5);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
