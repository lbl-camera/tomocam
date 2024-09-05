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
#include <cuda_runtime.h>

#include "dev_array.h"
#include "gpu/utils.cuh"

#include "potential_function.cuh"

namespace tomocam {
    namespace gpu {

        template <typename T>
        __global__ void tvd_update_kernel(DeviceMemory<T> model,
            DeviceMemory<T> objfn, float p, float sigma) {

            // thread ids
            int i = threadIdx.z;
            int j = threadIdx.y;
            int k = threadIdx.x;

            // global offsets
            int I = blockDim.z * blockIdx.z;
            int J = blockDim.y * blockIdx.y;
            int K = blockDim.x * blockIdx.x;

            // global ids
            int x = I + i;
            int y = J + j;
            int z = K + k;

            // last thread in the block
            dim3_t dims = objfn.dims();
            int imax = min(dims.x - I - 1, blockDim.z - 1);
            int jmax = min(dims.y - J - 1, blockDim.y - 1);
            int kmax = min(dims.z - K - 1, blockDim.x - 1);

            if ((x < dims.x) && (y < dims.y) && (z < dims.z)) {

                // size of the array
                dim3_t dims = objfn.dims();

                /* copy values into shared memory. */
                __shared__ T s_val[NX + 2][NY + 2][NZ + 2];

                // copy from global memory
                s_val[i + 1][j + 1][k + 1] = model.at(x, y, z);

                __syncthreads();
                /* copy ghost cells, on all 6 faces */
                // x = 0 face
                if (i == 0) s_val[i][j + 1][k + 1] = model.at(x - 1, y, z);

                // x = Nx-1 face
                if (i == imax)
                    s_val[i + 2][j + 1][k + 1] = model.at(x + 1, y, z);
                __syncthreads();

                if (j == 0) s_val[i + 1][j][k + 1] = model.at(x, y - 1, z);

                if (j == jmax)
                    s_val[i + 1][j + 2][k + 1] = model.at(x, y + 1, z);
                __syncthreads();

                if (k == 0) s_val[i + 1][j + 1][k] = model.at(x, y, z - 1);

                if (k == kmax)
                    s_val[i + 1][j + 1][k + 2] = model.at(x, y, z + 1);
                __syncthreads();

                /* copy ghost cells along 12 edges  */
                if (i == 0) {
                    if (j == 0) s_val[i][j][k + 1] = model.at(x - 1, y - 1, z);
                    if (j == jmax)
                        s_val[i][j + 2][k + 1] = model.at(x - 1, y + 1, z);
                }
                if (i == imax) {
                    if (j == 0)
                        s_val[i + 2][j][k + 1] = model.at(x + 1, y - 1, z);
                    if (j == jmax)
                        s_val[i + 2][j + 2][k + 1] = model.at(x + 1, y + 1, z);
                }
                __syncthreads();

                if (j == 0) {
                    if (k == 0) s_val[i + 1][j][k] = model.at(x, y - 1, z - 1);
                    if (k == kmax)
                        s_val[i + 1][j][k + 2] = model.at(x, y - 1, z + 1);
                }
                if (j == jmax) {
                    if (k == 0)
                        s_val[i + 1][j + 2][k] = model.at(x, y + 1, z - 1);
                    if (k == kmax)
                        s_val[i + 1][j + 2][k + 2] = model.at(x, y + 1, z + 1);
                }
                __syncthreads();

                // copy ghost-cells along y-direction
                if (k == 0) {
                    if (i == 0) s_val[i][j + 1][k] = model.at(x - 1, y, z - 1);
                    if (i == imax)
                        s_val[i + 2][j + 1][k] = model.at(x + 1, y, z - 1);
                }
                if (k == kmax) {
                    if (i == 0)
                        s_val[i][j + 1][k + 2] = model.at(x - 1, y, z + 1);
                    if (i == imax)
                        s_val[i + 2][j + 1][k + 2] = model.at(x + 1, y, z + 1);
                }
                __syncthreads();

                /*  copy  ghost cells along 16 corners */
                if (k == 0) {
                    if (j == 0) {
                        if (i == 0)
                            s_val[i][j][k] = model.at(x - 1, y - 1, z - 1);
                        if (i == imax) {
                            s_val[i + 2][j][k] = model.at(x + 1, y - 1, z - 1);
                        }
                    }
                    if (j == jmax) {
                        if (i == 0)
                            s_val[i][j + 2][k] = model.at(x - 1, y + 1, z - 1);
                        if (i == imax)
                            s_val[i + 2][j + 2][k] =
                                model.at(x + 1, y + 1, z - 1);
                    }
                }
                if (k == kmax) {
                    if (j == 0) {
                        if (i == 0)
                            s_val[i][j][k + 2] = model.at(x - 1, y - 1, z + 1);
                        if (i == imax)
                            s_val[i + 2][j][k + 2] =
                                model.at(x + 1, y - 1, z + 1);
                    }
                    if (j == jmax) {
                        if (i == 0)
                            s_val[i][j + 2][k + 2] =
                                model.at(x - 1, y + 1, z + 1);
                        if (i == imax)
                            s_val[i + 2][j + 2][k + 2] =
                                model.at(x + 1, y + 1, z + 1);
                    }
                }
                __syncthreads();

                T v = s_val[i + 1][j + 1][k + 1];
                T temp = 0.f;
                for (int ix = 0; ix < 3; ix++)
                    for (int iy = 0; iy < 3; iy++)
                        for (int iz = 0; iz < 3; iz++)
                            temp +=
                                weight(ix, iy, iz) *
                                d_pot_func(v - s_val[i + ix][j + iy][k + iz], p,
                                    sigma);
                objfn(x, y, z) += temp;
            }
        }

        template <typename T>
        void add_total_var(const DeviceArray<T> &sol, DeviceArray<T> &grad,
            float p, float sigma) {

            // CUDA kernel parameters
            Grid g(grad.dims());

            // update gradients inplace
            tvd_update_kernel<T>
                <<<g.blocks(), g.threads()>>>(sol, grad, p, sigma);
            SAFE_CALL(cudaGetLastError());
        }

        // instantiate the template
        template void add_total_var(const DeviceArray<float> &,
            DeviceArray<float> &, float, float);
        template void add_total_var(const DeviceArray<double> &,
            DeviceArray<double> &, float, float);

    } // namespace gpu
} // namespace tomocam
