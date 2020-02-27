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

    __global__ void fftshift1D_kernel(DeviceArray<cuComplex_t> arr) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;

        if (arr.valid(i, j, k)) {
            float a = powf(-1.f, k & 1);
            arr(i, j, k).x *= a;
            arr(i, j, k).y *= a;
        }
    }

    __global__ void fftshift2D_kernel(DeviceArray<cuComplex_t> arr) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;

        if (arr.valid(i, j, k)) {
            float a = powf(-1.f, (j + k) & 1);
            arr(i,j,k).x *= a; 
            arr(i,j,k).y *= a;
        }
    }

    __global__ void fftshift_center_kernel(DeviceArray<cuComplex_t> arr, float shift) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;

        dim3_t dims = arr.dims();
        if (arr.valid(i, j, k)) {
            float z = (float) k / (float) dims.z;
            float w = TWOPI * shift * z;
            arr(i,j,k)  = arr(i,j,k) * expf_j(-w);
        }
    }

    // multiply by -1^i
    void fftshift1D(DeviceArray<cuComplex_t> arr, cudaStream_t stream) {
        dim3 threads(1,1,256);
        dim3 tblocks = calcBlocks(arr.dims(), threads);
        fftshift1D_kernel<<< tblocks, threads, 0, stream >>>(arr);
    }

    // multiply by 2-D chessboard pattern
    void fftshift2D(DeviceArray<cuComplex_t> arr, cudaStream_t stream) {
        dim3 threads(1, 16, 16);
        dim3 tblocks = calcBlocks(arr.dims(), threads);
        fftshift2D_kernel<<< tblocks, threads, 0, stream>>>(arr);
    }

    // phase shift center
    void fftshift_center(DeviceArray<cuComplex_t> arr, float center, cudaStream_t stream) {
        dim3 threads(1, 1, 256);
        dim3 tblocks = calcBlocks(arr.dims(), threads);
        fftshift_center_kernel<<< tblocks, threads, 0, stream >>>(arr, center);
    }

    // undo phase shift center
    void ifftshift_center(DeviceArray<cuComplex_t> arr, float center, cudaStream_t stream) {
        dim3 threads(1, 1, 256);
        dim3 tblocks = calcBlocks(arr.dims(), threads);
        fftshift_center_kernel<<< tblocks, threads, 0, stream >>>(arr, -center);
    }

} // namespace tomocam
