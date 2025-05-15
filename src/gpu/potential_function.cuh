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
#include "gpu/utils.cuh"

#ifndef TOMOCAM_POTENTIAL_FUCTION__H
#define TOMOCAM_POTENTIAL_FUCTION__H

namespace tomocam {
    namespace gpu {
        const int NX = 1;
        const int NY = 16;
        const int NZ = 16;
        const float MRF_Q = 2.f;
        const float MRF_C = 0.001f;

        // clang-format off
        __device__ const float FILTER[3][3][3] = {{
                {0.0302, 0.0370, 0.0302},
                {0.0370, 0.0523, 0.0370},
                {0.0302, 0.0370, 0.0302}
            },
            {   {0.0370, 0.0523, 0.0370},
                {0.0523, 0.0000, 0.0523},
                {0.0370, 0.0523, 0.0370}
            },
            {   {0.0302, 0.0370, 0.0302},
                {0.0370, 0.0523, 0.0370},
                {0.0302, 0.0370, 0.0302}
            }
        };

        // clang-format on

        __deviceI__ float weight(int i, int j, int k) {
            return FILTER[i][j][k];
        }

        /*
         *            (|d| / sigma)^2
         *  f(d) =  -------------------
         *          c + (|d| / sigma)^(2-p)
         */
        template <typename T>
        __deviceI__ T potfunc(T delta, T p, T sigma) {
            auto y = abs(delta) / sigma;
            return pow(y, 2) / (MRF_C + pow(y, 2 - p));
        }

        template <typename T>
        __deviceI__ T d_potfunc(T delta, T p, T sigma) {

            auto x = abs(delta) / sigma;
            auto dx = 1.0 / sigma;
            if (x < 0) dx *= -1;

            // denominator
            auto den = MRF_C + pow(x, 2 - p);

            // first term
            auto t1 = 2 * x * dx / den;

            // second term
            auto t2 = - (2 - p) * pow(x, 3 - p) * dx / (den * den);
            if (isnan(t2)) printf("t2 = %g, den = %g\n", p, den);

            return t1 + t2;
        }

        __deviceI__ float d_pot_func(float delta, float MRF_P, float MRF_SIGMA) {

            float MRF_SIGMA_Q = powf(MRF_SIGMA, MRF_Q);
            float MRF_SIGMA_Q_P = powf(MRF_SIGMA, MRF_Q - MRF_P);

            float temp1 = powf(fabs(delta), MRF_Q - MRF_P) / MRF_SIGMA_Q_P;
            float temp2 = powf(fabs(delta), MRF_Q - 1);
            float temp3 = MRF_C + temp1;

            if (delta < 0.f) {
                return ((-1 * temp2 / (temp3 * MRF_SIGMA_Q)) *
                    (MRF_Q - ((MRF_Q - MRF_P) * temp1) / (temp3)));
            } else if (delta > 0.f) {
                return ((temp2 / (temp3 * MRF_SIGMA_Q)) *
                    (MRF_Q - ((MRF_Q - MRF_P) * temp1) / (temp3)));
            } else {
                return 0; // MRF_Q / (MRF_SIGMA_Q*MRF_C);
            }
        }

        /*Second Derivative of the potential function at zero */
        __deviceI__ float d2_pot_func_zero(float MRF_SIGMA) {
            float MRF_SIGMA_Q = powf(MRF_SIGMA, MRF_Q);
            return MRF_Q / (MRF_SIGMA_Q * MRF_C);
        }

    } // namespace gpu
} // namespace tomocam

#endif // TOMOCAM_POTENTIAL_FUCTION__H
