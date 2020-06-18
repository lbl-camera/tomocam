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

    __global__ 
    void fftshift1D_kernel(dev_arrayc arr) {
        int3 idx = Index3D();
        if (idx < arr.dims()) {
            float a = powf(-1.f, idx.z & 1);
            arr[idx] = arr[idx] * a;
        }
    }

    __global__ 
    void fftshift2D_kernel(dev_arrayc arr) {
        int3 idx = Index3D();
        if (idx < arr.dims()) {
            float a = powf(-1.f, (idx.y + idx.z) & 1);
            arr[idx] = arr[idx] * a;
        }
    }

    __global__ 
    void fftshift_center_kernel(dev_arrayc arr, float shift) {
        int3 idx = Index3D();
        dim3_t dims = arr.dims();
        if (idx < dims) {
            float z = (float) idx.z / dims.z;
            float w = TWOPI * shift * z;
            arr[idx] = arr[idx] * expf_j(-w);
        }
    }

    // multiply by -1^i
    void fftshift1D(dev_arrayc arr, cudaStream_t stream) {
        Grid grid(arr.dims());
        fftshift1D_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(arr);
    }

    // multiply by 2-D chessboard pattern
    void fftshift2D(dev_arrayc arr, cudaStream_t stream) {
        Grid grid(arr.dims());
        fftshift2D_kernel<<< grid.blocks(), grid.threads(), 0, stream>>>(arr);
    }

    // phase shift center
    void fftshift_center(dev_arrayc arr, float center, cudaStream_t stream) {
        Grid grid(arr.dims());
        fftshift_center_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(arr, center);
    }

    // undo phase shift center
    void ifftshift_center(dev_arrayc arr, float center, cudaStream_t stream) {
        Grid grid(arr.dims());
        fftshift_center_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(arr, -center);
    }
} // namespace tomocam
