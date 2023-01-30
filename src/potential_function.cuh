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
     *          1 + (|d| / sigma)^(q-p)
     */
    __devhstI__ float pot_func(float delta, float MRF_P, float MRF_SIGMA) {
        float g = fabs(delta) / MRF_SIGMA;
        float numer = powf(g, 2);
        float denom = (1 + powf(g, 2-MRF_P));
        return (numer/denom);
    }

    __devhstI__ float d_pot_func(float delta, float MRF_P, float MRF_SIGMA) {
        float g = fabs(delta) / MRF_SIGMA;
        float gprime = sgnf(delta) / MRF_SIGMA;

        float temp0 = powf(g, 2-MRF_P);
        float numer = g * gprime * (2 + MRF_P * temp0);
        float denom = powf(1 + temp0, 2);
        return (numer/denom);
    }

    /*Second Derivative of the potential function at zero */
    __devhstI__ float dd_pot_func0(float MRF_SIGMA) {
        return (2.f / MRF_SIGMA / MRF_SIGMA);
    }

    /*Second Derivative of the potential function */
    __devhstI__ float dd_pot_func(float delta, float MRF_P, float MRF_SIGMA) {
        float g = fabs(delta) / MRF_SIGMA;
        float gpsq = 1.f / powf(MRF_SIGMA, 2);
        float g_2mp = powf(g, 2-MRF_P);
        float numer1 = 2.f + MRF_P * (3.f - MRF_P)*g_2mp;
        float numer2 = (4.f - 2.f * MRF_P)*(2.f + MRF_P * g_2mp) * g_2mp;
        float denom1 = powf(1.f + g_2mp, 2);
        float denom2 = powf(1.f + g_2mp, 3);
        return (gpsq * ((numer1/denom1) - (numer2/denom2)));
    }

} // namespace tomocam


#endif // TOMOCAM_POTENTIAL_FUCTION__H
