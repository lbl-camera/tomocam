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
    void polar2cart_nufft(dev_arrayc input, dev_arrayc output,
            dev_arrayf angles, kernel_t kernel) {

        int3 idx = Index3D();
        dim3_t idims = input.dims();
        dim3_t odims = output.dims();

        if (idx < idims) {

            // polar coordinates
            float cen = (float) (idims.z) * 0.5;
            float ang = angles[idx.y];
            float y = (idx.z - cen) * sinf(ang) + cen;
            float z = (idx.z - cen) * cosf(ang) + cen;

            // indices where kerel is non-zero
            int iy    = max(kernel.imin(y), 0);
            int iymax = min(kernel.imax(y), odims.y - 1);
            int izmin = max(kernel.imin(z), 0);
            int izmax = min(kernel.imax(z), odims.z - 1);

            // value at (x, y)
            cuComplex_t val = input[idx];

            // convolve
            for (; iy < iymax; iy++) {
                cuComplex_t temp = val * kernel.weight(y-iy); 
                for (int iz = izmin; iz < izmax; iz++) {
                    cuComplex_t v = temp * kernel.weight(z-iz); 
                    atomicAdd(&output(idx.x, iy, iz).x, v.x);
                    atomicAdd(&output(idx.x, iy, iz).y, v.y);
                }
            }
        }
    }

    void polarsample_transpose(dev_arrayc input, dev_arrayc output,
        dev_arrayf angles, kernel_t kernel, cudaStream_t stream) {

        // cuda kernel params
        Grid grid(input.dims());

        // launch CUDA kernel
        polar2cart_nufft <<<grid.blocks(), grid.threads(), 0, stream>>> (input, output, angles, kernel);
    }
} // namespace tomocam
