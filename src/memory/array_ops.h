
#ifndef DIST_ARRAY_OPS__H
#define DIST_ARRAY_OPS__H

#include "memory/dist_array.h"
#include <cmath>
#include <execution>
#include <numeric>

namespace tomocam {

    namespace exe = std::execution;

    namespace array {
        // maximum
        template <typename T>
        T max(const DArray<T> &arr) {
            return *std::max_element(exe::par_unseq, arr.begin(), arr.end());
        }

        // minimum
        template <typename T>
        T min(const DArray<T> &arr) {
            return *std::min_element(exe::par_unseq, arr.begin(), arr.end());
        }

        // reduction sum
        template <typename T>
        T reduce_sum(const DArray<T> &arr) {
            return std::reduce(exe::par_unseq, arr.begin(), arr.end(), T(0));
        }

        // L1-norm
        template <typename T>
        T norm1(const DArray<T> &arr) {
            return std::transform_reduce(exe::par_unseq, arr.begin(), arr.end(),
                                         T(0), std::plus<T>(),
                                         [](T x) { return std::abs(x); });
        }

        // L2-norm
        template <typename T>
        T norm2(const DArray<T> &arr) {
            T sum_sq =
                std::transform_reduce(exe::par_unseq, arr.begin(), arr.end(), T(0),
                                      std::plus<T>(), [](T x) { return x * x; });
            return std::sqrt(sum_sq);
        }

        // dot product
        template <typename T>
        T dot(const DArray<T> &arr1, const DArray<T> &arr2) {
            if (arr1.dims() != arr2.dims()) {
                throw std::invalid_argument(
                    "Dot product error: Arrays must have the same dimensions.");
            }
            return std::transform_reduce(exe::par_unseq, arr1.begin(), arr1.end(),
                                         arr2.begin(), T(0));
        }
    } // namespace array

    // Binary operators for DArray + DArray
    template <typename T>
    DArray<T> operator+(const DArray<T> &lhs, const DArray<T> &rhs) {
        DArray<T> out = lhs.clone();
        out += rhs;
        return out;
    }

    template <typename T>
    DArray<T> operator-(const DArray<T> &lhs, const DArray<T> &rhs) {
        DArray<T> out = lhs.clone();
        out -= rhs;
        return out;
    }

    template <typename T>
    DArray<T> operator*(const DArray<T> &lhs, const DArray<T> &rhs) {
        DArray<T> out = lhs.clone();
        out *= rhs;
        return out;
    }

    template <typename T>
    DArray<T> operator/(const DArray<T> &lhs, const DArray<T> &rhs) {
        DArray<T> out = lhs.clone();
        out /= rhs;
        return out;
    }

    // Binary operators for DArray + scalar (right)
    template <typename T>
    DArray<T> operator+(const DArray<T> &lhs, const T &rhs) {
        DArray<T> out = lhs.clone();
        out += rhs;
        return out;
    }

    template <typename T>
    DArray<T> operator-(const DArray<T> &lhs, const T &rhs) {
        DArray<T> out = lhs.clone();
        out -= rhs;
        return out;
    }

    template <typename T>
    DArray<T> operator*(const DArray<T> &lhs, const T &rhs) {
        DArray<T> out = lhs.clone();
        out *= rhs;
        return out;
    }

    template <typename T>
    DArray<T> operator/(const DArray<T> &lhs, const T &rhs) {
        DArray<T> out = lhs.clone();
        out /= rhs;
        return out;
    }

    // Binary operators for scalar + DArray (left)
    template <typename T>
    DArray<T> operator+(const T &lhs, const DArray<T> &rhs) {
        DArray<T> out = rhs.clone();
        out += lhs;
        return out;
    }

    template <typename T>
    DArray<T> operator-(const T &lhs, const DArray<T> &rhs) {
        DArray<T> out = rhs.clone();
        out -= lhs;
        return out;
    }

    template <typename T>
    DArray<T> operator*(const T &lhs, const DArray<T> &rhs) {
        DArray<T> out = rhs.clone();
        out *= lhs;
        return out;
    }

    template <typename T>
    DArray<T> operator/(const T &lhs, const DArray<T> &rhs) {
        DArray<T> out = rhs.clone();
        out /= lhs;
        return out;
    }

} // namespace tomocam
#endif // DIST_ARRAY_OPS__H
