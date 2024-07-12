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

#include "gpu/utils.cuh"

namespace tomocam {
    namespace gpu {
        __device__ const double TWO_PI = 2 * CUDART_PI;

        // kernel to calculate grid-positions (single precision)
        template <typename T>
        __global__ void nugrid_kernel(int ncols, int nproj, T *x, T *y, const T *angles) {

            int jp = blockDim.x * blockIdx.x + threadIdx.x;
            int jc = blockDim.y * blockIdx.y + threadIdx.y;

            int center = ncols / 2;
            T s = static_cast<T>(TWO_PI) / ncols;
            if ((jp < nproj) && (jc < ncols)) {
                T r = s * static_cast<T>(jc - center);
                x[jp * ncols + jc] = r * cos(angles[jp]);
                y[jp * ncols + jc] = r * sin(angles[jp]);
            }
        }

        // wrapper to call nugrid_kernel
        template <typename T>
        void make_nugrid(int ncols, int nproj, T *x, T *y, const T *angles) {

            // copy angles to device
            T *d_angles;
            SAFE_CALL(cudaMalloc(&d_angles, sizeof(T) * nproj));
            SAFE_CALL(cudaMemcpy(d_angles, angles, sizeof(T) * nproj, cudaMemcpyHostToDevice));

            auto ceil = [](int a, int b) {
                if (a % b) return a / b + 1;
                else
                    return a / b;
            };

            // prepare of launch cuda-kernel
            int n1 = ceil(nproj, 32);
            int n2 = ceil(ncols, 32);
            dim3 threads(32, 32, 1);
            dim3 blocks(n1, n2, 1);

            // calculate grid-positions
            nugrid_kernel<T><<<blocks, threads>>>(ncols, nproj, x, y, d_angles);

            // free device angles
            SAFE_CALL(cudaFree(d_angles));
        }

        // explicit instantiation
        template void make_nugrid<float>(int, int, float *, float *,
            const float *);
        template void make_nugrid<double>(int, int, double *, double *,
            const double *);

    } // namespace gpu
} // namespace tomocam
