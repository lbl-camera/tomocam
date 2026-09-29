
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include "dist_array.h"

#ifndef UTILS__H
#define UTILS__H

class Timer {
  private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::milliseconds duration_;

  public:
    Timer() : duration_(0) {}

    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        duration_ =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    }

    void reset() { duration_ = std::chrono::milliseconds(0); }
    int ms() { return duration_.count(); }

    double seconds() { return static_cast<double>(duration_.count()) / 1000.0; }
};

class NPRandom {
  private:
    std::mt19937 gen_;

  public:
    NPRandom(unsigned int seed = 5489u) { gen_ = std::mt19937(seed); }

    template <typename T>
    T rand() {
        int a = gen_() >> 5;
        int b = gen_() >> 6;
        return static_cast<T>((a * 67108864.0 + b) / 9007199254740992.0);
    }

    template <typename T>
    std::vector<T> rand(size_t n) {
        std::vector<T> result(n);
        for (size_t i = 0; i < n; ++i) { result[i] = rand<T>(); }
        return result;
    }
};

// prints PASS/FAIL and returns the matching process exit code -- the
// convention every correctness test in this directory follows.
inline int report(bool ok) {
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}

// canonical relative-error metric used across the op-level tests:
// sqrt(||a - b||^2 / ||b||^2), since DArray::norm() already returns the
// sum of squares (not its square root).
template <typename T>
double rel_error(const tomocam::DArray<T> &a, const tomocam::DArray<T> &b) {
    double num = static_cast<double>((a - b).norm());
    double den = static_cast<double>(b.norm());
    return std::sqrt(num / std::max(den, 1e-30));
}

template <typename T>
tomocam::DArray<T> random_uniform(tomocam::dim3_t dims, T lo, T hi,
    unsigned seed) {
    tomocam::DArray<T> a(dims);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> u(lo, hi);
    for (uint64_t i = 0; i < a.size(); ++i) a[i] = u(rng);
    return a;
}

template <typename T>
tomocam::DArray<T> random_normal(tomocam::dim3_t dims, T mean, T stddev,
    unsigned seed) {
    tomocam::DArray<T> a(dims);
    std::mt19937 rng(seed);
    std::normal_distribution<T> n(mean, stddev);
    for (uint64_t i = 0; i < a.size(); ++i) a[i] = n(rng);
    return a;
}

#endif // UTILS__H
