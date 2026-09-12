// Shared harness for the manual bench_*.cpp benchmarks (not part of ctest):
// timing/Stats plumbing, a relative-error check, and CLI flag parsing that
// used to be duplicated near-verbatim across all 5 files.
#ifndef TOMOCAM_TEST_BENCH_SUPPORT__H
#define TOMOCAM_TEST_BENCH_SUPPORT__H

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include "dist_array.h"

namespace bench {

    struct Stats {
        double mean_ms;
        double min_ms;
        double gbytes_moved;
    };

    inline double now_ms() {
        using namespace std::chrono;
        return duration<double, std::milli>(
                   high_resolution_clock::now().time_since_epoch())
            .count();
    }

    // Fn returns elapsed ms for one call (no correctness value to track).
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

    // Fn returns {elapsed ms, a correctness value} for one call -- the
    // last rep's value is written to *last_value, if non-null.
    template <typename Fn, typename T>
    Stats time_it(Fn &&fn, int reps, double bytes_per_call, T *last_value) {
        fn(); // warmup, discarded
        double total = 0.0, best = 1e300;
        T last{};
        for (int i = 0; i < reps; i++) {
            auto [t, v] = fn();
            total += t;
            if (t < best) best = t;
            last = v;
        }
        if (last_value) *last_value = last;
        Stats s;
        s.mean_ms = total / reps;
        s.min_ms = best;
        s.gbytes_moved = bytes_per_call / 1e9;
        return s;
    }

    inline void print_row(const char *label, int chunks, const Stats &s) {
        double gbps = s.gbytes_moved / (s.min_ms / 1000.0);
        std::fprintf(stdout, "%-10s %6d %10.3f %10.3f %10.2f\n", label, chunks,
            s.mean_ms, s.min_ms, gbps);
        std::fflush(stdout);
    }

    // relative error between two arrays, e.g. raw vs. scheduled/chunked
    // output of the same op -- a correctness check, not a timing metric.
    template <typename T>
    T rel_error(const tomocam::DArray<T> &a, const tomocam::DArray<T> &b) {
        double num = 0.0, den = 0.0;
        for (uint64_t i = 0; i < a.size(); i++) {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            num += d * d;
            den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        }
        return static_cast<T>(std::sqrt(num / std::max(den, 1e-30)));
    }

    // scalar sibling of the DArray rel_error above, for benchmarks whose
    // correctness check is a single reduced value (function_value/2).
    template <typename T>
    T rel_error(T a, T b) {
        return std::abs(a - b) / std::max(std::abs(a), static_cast<T>(1e-30));
    }

    struct IntFlag {
        const char *name;
        int *value;
    };

    // parses "--name N" pairs into the bound targets; prints `usage` and
    // exits(0) on "--help". Unrecognized flags are silently ignored.
    inline void parse_int_flags(int argc, char **argv,
        std::initializer_list<IntFlag> flags, const char *usage) {
        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], "--help") == 0) {
                std::fprintf(stdout, "%s\n", usage);
                std::exit(0);
            }
            for (auto &f : flags) {
                if (std::strcmp(argv[i], f.name) == 0 && i + 1 < argc) {
                    *f.value = std::atoi(argv[++i]);
                    break;
                }
            }
        }
    }

} // namespace bench

#endif // TOMOCAM_TEST_BENCH_SUPPORT__H
