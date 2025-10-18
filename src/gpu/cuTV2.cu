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
        __global__ void qGGMRF_kernel(DeviceMemory<T> f_ext,
            DeviceMemory<T> g, T sigma, T p) {


                // get 3D index
                __shared__ T s_val[NX + 2][NY + 2][NZ + 2];

                auto idx = Index3D();
                if (idx <  g.dims()) {

                    // last block with shift less thet block size
                    int shiftx = min(blockDim.z, g.dims().x - blockIdx.z * blockDim.z);
                    int shifty = min(blockDim.y, g.dims().y - blockIdx.y * blockDim.y);
                    int shiftz = min(blockDim.x, g.dims().z - blockIdx.x * blockDim.x);
                    
                    /* copy values into shared memory. */
                    for (int i = threadIdx.z; i < NX + 2; i += shiftx) {
                        for (int j = threadIdx.y; j < NY + 2; j += shifty) {
                            for (int k = threadIdx.x; k < NZ + 2; k += shiftz) {
                                int x = (int) (blockIdx.z * blockDim.z) + i - 1;
                                int y = (int) (blockIdx.y * blockDim.y) + j - 1;
                                int z = (int) (blockIdx.x * blockDim.x) + k - 1;
                                s_val[i][j][k] = f_ext.at(x, y, z);
                            }
                        }
                    }
                    __syncthreads();

                    // compute the qGGMRF contribution
                    T v = s_val[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1];
                    T temp = 0.f;
                    for (int ix = 0; ix < 3; ix++) {
                        for (int iy = 0; iy < 3; iy++) {
                            for (int iz = 0; iz < 3; iz++) {
                                auto delta = v - s_val[threadIdx.z + ix][threadIdx.y + iy][threadIdx.x + iz];
                                auto tv = d_pot_func(delta, p, sigma);
                                temp += weight(ix, iy, iz) * tv;
                            }
                        }
                    }
                    g[idx] += temp;
                }
                __syncthreads();
        }

        template <typename T>
        void add_total_var2(const DeviceArray<T> &sol, DeviceArray<T> &grad, T sigma, T p) {

            // data size
            auto dims = grad.dims();
            // CUDA kernel parameters
            dim3 block(NZ, NY, NX);
            dim3 grid;
            grid.x = (dims.z + NZ - 1) / NZ;
            grid.y = (dims.y + NY - 1) / NY;
            grid.z = (dims.x + NX - 1) / NX;

            // update gradients inplace
            qGGMRF_kernel<T><<<grid, block>>>(sol, grad, sigma, p);
            SAFE_CALL(cudaGetLastError());
        }

        // instantiate the template
        template void add_total_var2(const DeviceArray<float> &,
            DeviceArray<float> &, float, float);
        template void add_total_var2(const DeviceArray<double> &,
            DeviceArray<double> &, double, double);

    } // namespace gpu
} // namespace tomocam
