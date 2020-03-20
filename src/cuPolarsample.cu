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
    void cart2polar_nufft(dev_arrayc input, dev_arrayc output, dev_arrayf angles, kernel_t kernel) {

        // get global index
        int3 idx = Index3D();
        dim3_t idims = input.dims();
        dim3_t odims = output.dims();

        if (idx < odims) {

            // polar coordinates
            float cen = (float) (odims.z) * 0.5;
            float ang = angles[idx.y];
            float y = (idx.z - cen) * cosf(ang) + cen;
            float z = (idx.z - cen) * sinf(ang) + cen;

            // get min and max of non-zero kernel
            int iy    = max(kernel.imin(y), 0);
            int iymax = min(kernel.imax(y), idims.y - 1);
            int izmin = max(kernel.imin(z), 0);
            int izmax = min(kernel.imax(z), idims.z - 1);

            for (; iy < iymax; iy++) {
                float wy = kernel.weight(y - iy);
                for (int iz = izmin; iz < izmax; iz++) {
                    float wz = kernel.weight(z - iz); 
                    output[idx] = output[idx] + (input(idx.x, iy, iz) * wy * wz);
                }
            }
        }
    }

    void polarsample(dev_arrayc input, dev_arrayc output, dev_arrayf angles,
            kernel_t kernel, cudaStream_t stream) {
        // parallelize over sinogram
        Grid grid(output.dims());
        cart2polar_nufft<<< grid.blocks(), grid.threads(), 0, stream>>>(input, output, angles, kernel);
    }
} // namespace tomocam
