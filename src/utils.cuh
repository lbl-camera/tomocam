/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley National
 * Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
 *  Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef TOMOCAM_UTILS__CUH
#define TOMOCAM_UTILS__CUH

#include <cuda.h>
#include "common.h"

#define CUDAFY __forceinline__ __device__ __host__
namespace tomocam {

    inline int idiv (int a, int b) {
        if (a % b) return (a / b + 1);
        else return (a / b);
    }

    inline dim3 calcBlocks(dim3_t dims, dim3 thrds) {
        return dim3(idiv(dims.x, thrds.x), idiv(dims.y, thrds.y), idiv(dims.z, thrds.z));
    }

    /* add complex to a complex */
    CUDAFY cuComplex_t operator+(cuComplex_t a, cuComplex_t b) {
        return make_cuFloatComplex(a.x + b.x, a.y + b.y);
    }

    /* multiply complex with a float */
    CUDAFY cuComplex_t operator*(cuComplex_t a, float b) {
        return make_cuFloatComplex(a.x * b, a.y * b);
    }
    CUDAFY cuComplex_t operator*(float b, cuComplex_t a) {
        return make_cuFloatComplex(a.x * b, a.y * b);
    }

    /* multiply complex with a complex */
    CUDAFY cuComplex_t operator*(cuComplex_t a, cuComplex_t b) { return cuCmulf(a, b); }

    CUDAFY cuComplex_t expf_j(const float arg) {
        float sin, cos;
        sincosf(arg, &sin, &cos);
        return make_cuFloatComplex(cos, sin);
    }

} // namespace tomocam
#endif // TOMOCAM_UTILS__CUH
