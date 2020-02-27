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

#include <cuda_runtime.h>

#include "dev_array.h"
#include "kernel.h"
#include "types.h"
#include "utils.cuh"

namespace tomocam {

    __global__ 
    void polar2cart_nufft(DeviceArray<cuComplex_t> input, 
        DeviceArray<cuComplex_t> output,
        DeviceArray<float> angles, kernel_t kernel) {

        // get global index
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
 
        dim3_t idims = input.dims();
        dim3_t odims = output.dims();

        if ((i < idims.x) && (j < idims.y) && (k < idims.z)) {

            // polar coordinates
            float cen = (float) (idims.z) * 0.5;
            float ang = angles[j];
            float y = (k - cen) * sinf(ang) + cen;
            float z = (k - cen) * cosf(ang) + cen;

            // indices where kerel is non-zero
            int iy    = max(kernel.imin(y), 0);
            int iymax = min(kernel.imax(y), odims.y - 1);
            int izmin = max(kernel.imin(z), 0);
            int izmax = min(kernel.imax(z), odims.z - 1);

            // value at (x, y)
            cuComplex_t pValue = input(i, j, k);

            // convolve
            for (; iy < iymax; iy++) {
                cuComplex_t temp = pValue * kernel.weight(y-iy); 
                for (int iz = izmin; iz < izmax; iz++) {
                    cuComplex_t v = temp * kernel.weight(z-iz); 
                    atomicAdd(&output(i, iy, iz).x, v.x);
                    atomicAdd(&output(i, iy, iz).y, v.y);
                }
            }
        }
    }

    void polarsample_transpose(DeviceArray<cuComplex_t> input, DeviceArray<cuComplex_t> output,
        DeviceArray<float> angles, kernel_t kernel, cudaStream_t stream) {

        // cuda kernel params
        dim3 threads(1, 1, 256);
        dim3 tblocks = calcBlocks(input.dims(), threads);

        // launch CUDA kernel
        polar2cart_nufft <<<tblocks, threads, 0, stream>>> (input, output, angles, kernel);
    }
} // namespace tomocam
