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

#include <omp.h>

#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "types.h"

namespace tomocam {

    const float weight [3][3][3] = {
            {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}},
            {{0.037, 0.0523, 0.037}, {0.0523, 0., 0.0523}, {0.037, 0.0523, 0.037}},
            {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}}
        };

    const float MRF_Q = 2.f;
    const float MRF_C = 0.001;

    float d_potfun(float d, float sigma, float p) {
        float delta     = std::abs(d);
        float sigma_q   = std::pow(sigma, MRF_Q);
        float sigma_q_p = std::pow(sigma, MRF_Q - p);

        float temp1 = std::pow(delta, MRF_Q - p) / sigma_q_p;
        float temp2 = std::pow(delta, MRF_Q - 1);
        float temp3 = MRF_C + temp1;

        if (delta < 0.f) 
            return ((-temp2 / (temp3 * sigma_q)) * (MRF_Q - ((MRF_Q - p) * temp1) / temp3));
        else if (delta > 0.f) 
            return ((temp2 / (temp3 * sigma_q)) * (MRF_Q - ((MRF_Q - p) * temp1) / temp3));
        else 
            return 0; 
    }

    // calculate contraints on CPU 
    void cpuTotalVar(DArray<float> &input, DArray<float> &output, float sigma, float mrf_p) {

        // dims
        dim3_t dims = output.dims();
        auto p = input.create_partitions(1)[0];

        // write out for clarity
        int nslc = dims.x;
        int nrow = dims.y;
        int ncol = dims.z;

        // leave 4 theads for gpu-related calls
        int max_threads = omp_get_max_threads();
        if (max_threads >= 20) max_threads -= 4;
        #pragma omp parallel for num_threads(max_threads);
        for (int i = 0; i < nslc; i++) {
            for (int j = 0; j < nrow; j++) {
                for (int k = 0; k < ncol; k++) {
                    float u = p(i, j, k);
                    float v = 0.f;
                    for (int z = -1; z < 2; z++) {
                        for (int y = -1; y < 2; y++) {
                            for (int x = -1; x < 2; x++) {
                                float d = u - p.padded(i+z, j+y, k+x);
                                v += weight[z][y][x] * d_potfun(d, sigma, mrf_p);
                            }
                        }
                    }
                }
            }
        }
    }
} // namespace tomocam
