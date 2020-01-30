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
    void polar2cart_nufft(int3 idims, int3 odims, cuComplex_t *input, float *angles,
        kernel_t kernel, cuComplex_t *output) {

        // get global index
        int gid = blockDim.x * blockIdx.x + threadIdx.x;
 
        int IMAX = idims.x * idims.y * idims.z;
        if (gid <  IMAX ) {
            int islc = gid / idims. y / idims.z;
            int iloc = gid % (idims.y * idims.z);
            int iang = iloc / idims.z;
            int ipos = iloc % idims.z;

            // copy kernel to shared memory
            extern __shared__ float shamem_kfunc[];
            int niters = kernel.size() / blockDim.x;
            int nextra = kernel.size() % blockDim.x;

            size_t offset = 0;
            for (int j = 0; j < niters; j++) {
                offset = j * blockDim.x;
                shamem_kfunc[threadIdx.x + offset] = kernel.d_array()[threadIdx.x + offset];
            }
            if ((nextra > 0) && (threadIdx.x < nextra))
                shamem_kfunc[threadIdx.x + offset] = kernel.d_array()[threadIdx.x + offset];

            // polar coordinates
            float c = (float) (idims.z) * 0.5;
            float a = angles[iang];
            float x = (ipos - c) * cosf(a) + c;
            float y = (ipos - c) * sinf(a) + c;

            // value at (x, y)
            cuComplex_t pValue = input[gid];

            int iy    = max(kernel.imin(y), 0);
            int iymax = min(kernel.imax(y), odims.y - 1);
            int ixmin = max(kernel.imin(x), 0);
            int ixmax = min(kernel.imax(x), odims.z - 1);

            for (; iy < iymax; iy++) {
                cuComplex_t temp = pValue * kernel.weight(y, iy, shamem_kfunc);
                for (int ix = ixmin; ix < ixmax; ix++) {
                    cuComplex_t v = temp * kernel.weight(x, ix, shamem_kfunc);
                    int idx = islc * odims.y * odims.z + iy * odims.z + ix;
                    atomicAdd(&output[idx].x, v.x);
                    atomicAdd(&output[idx].y, v.y);
                }
            }
        }
    }

    void polarsample_transpose(cuComplex_t *input, cuComplex_t *output, dim3_t idims, dim3_t odims,
        DeviceArray<float> angles, kernel_t kernel, cudaStream_t stream) {

        // polar-coordinates
        float *d_angles = angles.d_array();

        // input and output dimensions
        int3 d_idims = make_int3(idims.x, idims.y, idims.z);
        int3 d_odims = make_int3(odims.z, odims.y, odims.z);
        int kdims     = kernel.size();

        // cuda kernel params
        int nmax = idims.x * idims.y * idims.z;
        int nthread = 256;
        int tblocks  = nmax / nthread + 1;

        // launch CUDA kernel
        polar2cart_nufft <<<tblocks, nthread, kdims * sizeof(float), stream>>> (
            d_idims, d_odims, input, d_angles, kernel, output);
    }
} // namespace tomocam
