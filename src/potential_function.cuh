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

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "utils.cuh"

#ifndef TOMOCAM_POTENTIAL_FUCTION__H
#define TOMOCAM_POTENTIAL_FUCTION__H

namespace tomocam {

    const int NX = 1;
    const int NY = 16;
    const int NZ = 16;
    const float MRF_Q = 2.f;
    const float MRF_C = 0.0001f;
||||||| fac178d
    const float MRF_Q = 2.f;
    const float MRF_C = 0.001f;
=======
    const float MRF_C = 0.0001f;
>>>>>>> c4669c5e617f9da04c6a3a5b42579aa14a661f54

    __device__ const float FILTER[3][3][3] = {
        {{0.0302, 0.0370, 0.0302}, {0.0370, 0.0523, 0.0370}, {0.0302, 0.0370, 0.0302}},
        {{0.0370, 0.0523, 0.0370}, {0.0523, 0.0000, 0.0523}, {0.0370, 0.0523, 0.0370}},
        {{0.0302, 0.0370, 0.0302}, {0.0370, 0.0523, 0.0370}, {0.0302, 0.0370, 0.0302}}};

    __deviceI__ float weight(int i, int j, int k) { return FILTER[i][j][k]; }

    __devhstI__ float sgnf(float v) {
        float u = fabs(v);
        if (u > 0) return (v/u);
        else return 0;
    }

    /*
     *            (|d| / sigma)^q
     *  f(d) =  -------------------
     *          c + (|d| / sigma)^(q-p)
     */
    __devhstI__ float pot_func(float delta, float p, float sigma) {
        float c = MRF_C;
        float g = fabs(delta) / sigma;
        float numer = powf(g, 2.f);
        float denom = (c + powf(g, 2.f - p));
        return (numer/denom);
    }

    __devhstI__ float d_pot_func(float delta, float p, float sigma) {
        float c = MRF_C;
        float g = fabs(delta) / sigma;
        float gprime = sgnf(delta) / sigma;

        float temp0 = powf(g, 2.f - p);
        float numer = g * gprime * (2.f * c + p * temp0);
        float denom = powf(c + temp0, 2.f);
        return (numer/denom);
    }

    /*Second Derivative of the potential function at zero */
    __devhstI__ float dd_pot_func0(float sigma) {
        return (2.f / sigma / sigma / MRF_C);
    }
} // namespace tomocam


#endif // TOMOCAM_POTENTIAL_FUCTION__H
