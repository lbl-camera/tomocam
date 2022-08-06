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

#include <cuda.h>
#include <math_constants.h>
#include "utils.cuh"

namespace tomocam {
    __device__ const float TWO_PI = 2 * CUDART_PI_F;

    __global__ void nufft_grid_kernel(int ncols, int nproj, 
            float *x, float *y, float *angles, float center) {

        int ic = blockDim.x * blockIdx.x + threadIdx.x;
        int jp = blockDim.y * blockIdx.y + threadIdx.y;
        float s = TWO_PI / ncols;
        if ((ic < ncols) && (jp < nproj)) {
            float r = s * (float) (ic - center);
            x[jp * ncols + ic] = r * cos(angles[jp]);
            y[jp * ncols + ic] = r * sin(angles[jp]);
        }
    }

    void nufft_grid(int ncols, int nproj, float *x, float *y,
            float *angles, float center) {
        // copy angles to device
        float *d_angles;
        SAFE_CALL(cudaMalloc(&d_angles, sizeof(float) * nproj));
        SAFE_CALL(cudaMemcpy(d_angles, angles, sizeof(float) * nproj, cudaMemcpyHostToDevice));

        // prepare of launch cuda-kernel
        int n1 = ceili(ncols, 16);
        int n2 = ceili(nproj, 16);
        dim3 threads(16, 16, 1);
        dim3 blocks(n1, n2, 1);

        // calculate grid-positions
        nufft_grid_kernel <<< blocks, threads >>> (ncols, nproj, x, y, d_angles, center);

        // free device angles
        SAFE_CALL(cudaFree(d_angles));
    }
}
