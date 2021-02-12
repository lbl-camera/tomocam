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
    const float MRF_C = 0.001f;

    __device__ const float FILTER[3][3][3] = {
        {{0.0302, 0.0370, 0.0302}, {0.0370, 0.0523, 0.0370}, {0.0302, 0.0370, 0.0302}},
        {{0.0370, 0.0523, 0.0370}, {0.0532, 0.0000, 0.0523}, {0.0370, 0.0523, 0.0370}},
        {{0.0302, 0.0370, 0.0302}, {0.0370, 0.0523, 0.0370}, {0.0302, 0.0370, 0.0302}}};

    __deviceI__ float weight(int i, int j, int k) { return FILTER[i][j][k]; }

    /*
     *            (|d| / sigma)^q
     *  f(d) =  -------------------
     *          c + (|d| / sigma)^(q-p)
     */
    __deviceI__ float pot_func(float delta, float MRF_P, float MRF_SIGMA) {
        return ((powf(fabs(delta) / MRF_SIGMA, MRF_Q)) / (MRF_C + powf(fabs(delta) / MRF_SIGMA, MRF_Q - MRF_P)));
    }

    __deviceI__ float d_pot_func(float delta, float MRF_P, float MRF_SIGMA) {
        float MRF_SIGMA_Q = powf(MRF_SIGMA, MRF_Q);
        float MRF_SIGMA_Q_P = powf(MRF_SIGMA, MRF_Q - MRF_P);

        float temp1 = powf(fabs(delta), MRF_Q - MRF_P) / MRF_SIGMA_Q_P;
        float temp2 = powf(fabs(delta), MRF_Q - 1);
        float temp3 = MRF_C + temp1;

        if (delta < 0.0) {
            return ((-1 * temp2 / (temp3 * MRF_SIGMA_Q)) * (MRF_Q - ((MRF_Q - MRF_P) * temp1) / (temp3)));
        } else if (delta > 0.0) {
            return ((temp2 / (temp3 * MRF_SIGMA_Q)) * (MRF_Q - ((MRF_Q - MRF_P) * temp1) / (temp3)));
        } else {
            return 0; // MRF_Q / (MRF_SIGMA_Q*MRF_C);
        }
    }

    /*Second Derivative of the potential function at zero */
    __deviceI__ float d2_pot_func_zero(float MRF_SIGMA) {
        float MRF_SIGMA_Q = powf(MRF_SIGMA, MRF_Q);
        return MRF_Q / (MRF_SIGMA_Q * MRF_C);
    }

} // namespace tomocam


#endif // TOMOCAM_POTENTIAL_FUCTION__H
