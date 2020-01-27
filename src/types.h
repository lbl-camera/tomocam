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

#ifndef TOMOCAM_TYPES__H
#define TOMOCAM_TYPES__H

#include <cmath>
#include <complex>
#include <cuComplex.h>
#include <cuda.h>

namespace {
    typedef std::complex<float> complex_t;
    typedef cuFloatComplex cuComplex_t;

    __device__ __host__  cuComplex_t operator*(cuComplex_t a, float b) {
        return make_cuFloatComplex(a.x * b, a.y * b);
    }
    __device__ __host__  cuComplex_t operator*(float b, cuComplex_t a) {
        return make_cuFloatComplex(a.x * b, a.y * b);
    }

    __device__ __host__  cuComplex_t operator*(cuComplex_t a, cuComplex_t b) { return cuCmulf(a, b); }

    __device__ __host__  cuComplex_t expf_j(const float arg) {
        float sin, cos;
        sincosf(arg, &sin, &cos);
        return make_cuFloatComplex(cos, sin);
    }

} // namespace

#endif // TOMOCAM_TYPES__H
