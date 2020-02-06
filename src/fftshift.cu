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

#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "fft.h"
#include "utils.cuh"

namespace tomocam {

    __device__ const float TWOPI = 6.283185307179586;

    __global__ void fftshift1D_kernel(cuComplex_t *arr, size_t len, size_t batches) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < len * batches) {
            int j   = i % len;
            float a = powf(-1.f, j & 1);
            arr[i].x *= a;
            arr[i].y *= a;
        }
    }

    __global__ void fftshift2D_kernel(cuComplex_t *arr, size_t nrow, size_t ncol, size_t batches) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        if ((i < batches) && (j < nrow) && (k < ncol)) {
            int gid = i * nrow * ncol + j * ncol + k;
            float a = powf(-1.f, (j + k) & 1);
            arr[gid].x *= a; 
            arr[gid].y *= a;
        }
    }

    __global__ void fftshift_center_kernel(cuComplex_t *arr, size_t len, size_t batches, float shift) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < len * batches) {
            float k = (float)(i % len) / (float) len;
            float w = TWOPI * shift * k;
            arr[i]  = arr[i] * expf_j(-w);
        }
    }

    // multiply by -1^i
    void fftshift1D(cuComplex_t *arr, dim3_t dims, cudaStream_t stream) {
        size_t nelem = dims.x * dims.y * dims.z;
        dim3 threads(256, 1, 1);
        dim3 tblocks(nelem / threads.x + 1, 1, 1);
        fftshift1D_kernel<<< tblocks, threads, 0, stream >>>(arr, dims.z, dims.x * dims.y);
    }

    // multiply by 2-D chessboard pattern
    void fftshift2D(cuComplex_t *arr, dim3_t dims, cudaStream_t stream) {
        dim3 threads(1, 16, 16);
        dim3 tblocks = calcBlocks(dims, threads);
        fftshift2D_kernel<<< tblocks, threads, 0, stream>>>(arr, dims.y, dims.z, dims.x);
    }

    // phase shift center
    void fftshift_center(cuComplex_t *arr, dim3_t dims, float center, cudaStream_t stream) {
        size_t nelem = dims.x * dims.y * dims.z;
        dim3 threads(256, 1, 1);
        dim3 tblocks(nelem / threads.x + 1, 1, 1);
        fftshift_center_kernel<<< tblocks, threads, 0, stream >>>(arr, dims.z, dims.x * dims.y, center);
    }

    // undo phase shift center
    void ifftshift_center(cuComplex_t *arr, dim3_t dims, float center, cudaStream_t stream) {
        size_t nelem = dims.x * dims.y * dims.z;
        dim3 threads(256, 1, 1);
        dim3 tblocks(nelem / threads.x + 1, 1, 1);
        fftshift_center_kernel<<< tblocks, threads, 0, stream >>>(arr, dims.z, dims.x * dims.y, -center);
    }

} // namespace tomocam
