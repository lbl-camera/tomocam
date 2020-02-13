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
    void cart2polar_nufft(dim3_t idims, dim3_t odims, cuComplex_t *input, 
        DeviceArray<float> angles, kernel_t kernel, cuComplex_t *output) {

        // get global index
        int gid = blockDim.x * blockIdx.x + threadIdx.x;
        int nmax = odims.x * odims.y * odims.z;
        if (gid < nmax) {
            int islc = gid / (odims.y * odims.z);
            int iloc = gid % (odims.y * odims.z);
            int iang = iloc / odims.z;
            int ipos = iloc % odims.z;

            // polar coordinates
            float c = (float) (odims.z) * 0.5;
            float a = angles[iang];
            float x = (ipos - c) * cosf(a) + c;
            float y = (ipos - c) * sinf(a) + c;

            // get min and max of non-zero kernel
            int iy    = max(kernel.imin(y), 0);
            int iymax = min(kernel.imax(y), idims.y - 1);
            int ixmin = max(kernel.imin(x), 0);
            int ixmax = min(kernel.imax(x), idims.z - 1);

            for (; iy < iymax; iy++) {
                float wy = kernel.weight(y - iy);
                for (int ix = ixmin; ix < ixmax; ix++) {
                    int idx = islc * idims.y * idims.z + iy * idims.z + ix;
                    float wx = kernel.weight(x - ix); 
                    output[gid] = output[gid] + input[idx] * wx * wy; 
                }
            }
        }
    }

    void polarsample(cuComplex_t *input, cuComplex_t *output, dim3_t idims, dim3_t odims, 
        DeviceArray<float> angles, kernel_t kernel, cudaStream_t stream) {

        // cuda kernel params
        int nmax = odims.x * odims.y * odims.z;
        int nthread = 256;
        int tblocks = idiv(nmax, nthread);

        // launch CUDA kernel
        cart2polar_nufft<<< tblocks, nthread, 0, stream>>>(
            idims, odims, input, angles, kernel, output); 
    }
} // namespace tomocam
