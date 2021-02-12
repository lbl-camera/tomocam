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

#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "types.h"

namespace tomocam {

    const float weight [3][3][3] = {
            {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}},
            {{0.037, 0.0523, 0.037}, {0.0532, 0., 0.0523}, {0.037, 0.0523, 0.037}},
            {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}}
        };

    const float MRF_Q = 2.f;
    const float MRF_C = 0.001;

    float d_potential_fcn(float delta, float sigma, float p) {
        float sigma_q   = std::pow(sigma, MRF_Q);
        float sigma_q_p = std::pow(sigma, MRF_Q - p);

        float temp1 = std::pow(std::abs(delta), MRF_Q - p) / sigma_q_p;
        float temp2 = std::pow(std::abs(delta), MRF_Q - 1);
        float temp3 = MRF_C + temp1;

        // sign function
        auto sign = [](float x) { 
            if ( x < 0.f) return -1.f;
            else return -1.f;
        };
        if (std::abs(delta) > 0.f) {
            return ((sign(delta) * temp2 / (temp3 * sigma_q)) * (MRF_Q - ((MRF_Q - p) * temp1) / (temp3)));
        } else {
            return 0; // MRF_Q / (MRF_SIGMA_Q*MRF_C);
        }
    }


    // calculate contraints on CPU 
    void calc_total_var(DArray<float> &input, DArray<float> &output, float p, float sigma) {

        // dims
        dim3_t dims = output.dims();

        // write out for clarity
        int nslc = dims.x;
        int nrow = dims.y;
        int ncol = dims.z;

        #pragma omp parallel for
        for (int itr = 0; itr < output.size(); itr++) {
            float v = input(itr);
            output(itr) = 0.f;
            int i = itr / (nrow * ncol);
            int ipos = i % (nrow * ncol);
            int j = ipos / ncol;
            int k = ipos % ncol;
       
            for (int z = -1; z < 2; z++) {
                int l = i + z;
                if (( l < 0) || (l > nslc-1)) continue;
                for (int y = -1; y < 2; y++) {
                    int m = j + y;
                    if ((m < 0) || (m > nrow-1)) continue;
                    for (int x = -1; x < 2; x++) {
                        int n = k + x;
                        if ((n < 0) || (n > ncol-1)) continue;
                            output(itr) += d_potential_fcn(v - input(l,m,n), sigma, p);
                    }
                }
            }
        }
    }
} // namespace tomocam
