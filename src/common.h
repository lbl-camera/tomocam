/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <format>
#include <iostream>
#include <string>

#ifndef TOMOCAM_COMMON__H
#define TOMOCAM_COMMON__H

namespace tomocam {

    struct ReconParams {
        size_t max_iters = 100;  // outer iteration count (nagopt / split-Bregman)
        size_t inner_iters = 1;  // CG inner-loop cap (cgsolver, within split-Bregman)
        double tol;
        double xtol;
        double mu = 10.0;     // split-Bregman quadratic penalty weight
        double lambda = 0.1;  // TV shrinkage weight
        double sigma = 500.0; // qGGMRF parameter
    };

    struct OutputParams {
        std::string filename;
        std::string format;

        std::string insert_rank(size_t rank) {
            // insert left-zero-padded rank into filename
            std::string s = std::format("{:05d}", rank);
            size_t dot = filename.find_last_of(".");
            std::string basename = filename.substr(0, dot);
            std::string ext = filename.substr(dot);
            return basename + "_" + s + ext;
        }
    };

    enum class PadType { LEFT, RIGHT, SYMMETRIC };

    struct dim3_t {
        int x, y, z;
        dim3_t() : x(1), y(1), z(1) {}
        dim3_t(int d0, int d1, int d2) : x(d0), y(d1), z(d2) {}

        bool operator==(const dim3_t &other) const {
            if ((x == other.x) && (y == other.y) && (z == other.z))
                return true;
            else
                return false;
        }

        bool operator!=(const dim3_t &other) const {
            if ((x == other.x) && (y == other.y) && (z == other.z))
                return false;
            else
                return true;
        }

        dim3_t operator-(const dim3_t &other) const {
            dim3_t v(x - other.x, y - other.y, z - other.z);
            return v;
        }

        dim3_t operator+(const dim3_t &other) const {
            dim3_t v(x + other.x, y + other.y, z + other.z);
            return v;
        }

        dim3_t operator*(int scalar) const {
            dim3_t v(x * scalar, y * scalar, z * scalar);
            return v;
        }

        bool isNULL() const {
            if ((x == 0) && (y == 0) && (z == 0))
                return true;
            else
                return false;
        }

#ifdef __NVCC__
        __host__ __device__ dim3_t operator=(const int3 &rhs) {
            x = rhs.x;
            y = rhs.y;
            z = rhs.z;
            return *this;
        }

        __host__ __device__ operator int3() const { return make_int3(x, y, z); }
#endif // __NVCC__
    };

    inline dim3_t operator*(int scalar, const dim3_t &v) { return v * scalar; }

    inline std::ostream &operator<<(std::ostream &os, const dim3_t &v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
} // namespace tomocam
#endif // TOMOCAM_COMMON__H
