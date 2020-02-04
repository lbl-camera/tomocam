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
#include "types.h"
#include "utils.cuh"

namespace tomocam {

    __device__ 
    float kaiser_fourier_trans(float x, float W, float beta) {
        const float PI = 3.14159265359f;
        float t1 = powf(beta, 2) - powf(x * W * PI, 2);

        if (t1 > 0 ) { 
            t1 = sqrtf(t1);
            float t2 = W / cyl_bessel_i0(beta);
            return (t2 * sinhf(t1)/t1);
        } 
        else return 1.f;
    }
        
    __global__ 
    void deapodize2d_kernel(cuComplex_t * arr, float W, float beta, int2 grid, int slcs) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;

        int nrows  = grid.x;
        int ncols  = grid.y;
        float nmax = (float) ncols;
        float cen  = (float) (nmax / 2);

        if ((i < slcs) && (j < nrows) && (k < ncols)) {
            float y = (j - cen) / nmax; 
            float x = (k - cen) / nmax;
            float wx = kaiser_fourier_trans(x, W, beta);
            float wy = kaiser_fourier_trans(y, W, beta);
            int gid = i * nrows * ncols + j * ncols + k;
            arr[gid].x = arr[gid].x / wx / wy;
        }
    }


    __global__ 
    void deapodize1d_kernel(cuComplex_t * arr, float W, float beta, int cols, int rows) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        
        float nmax = (float) cols;
        float cen  = (float) (cols / 2);
        if (i < rows * cols) {
            int j = i % rows;
            float x = (j - cen) /  nmax;
            float wx  = kaiser_fourier_trans(x, W, beta);
            arr[i].x = arr[i].x / wx;
        }
    }


    // kernel wrappers 
    void deApodize2D(cuComplex_t * arr, dim3_t dims, float W, float beta, cudaStream_t stream) {
        int slcs  = dims.x;
        int2 grid = make_int2(dims.y, dims.z);
        dim3 threads(1, 16, 16);
   
        dim3 tblocks = calcBlocks(dims, threads);
        deapodize2d_kernel <<< tblocks, threads, 0, stream >>> (arr, W, beta, grid, slcs);
    }


    void deApodize1D(cuComplex_t * arr, dim3_t dims, float W, float beta, cudaStream_t stream) {
        int size = dims.x * dims.y * dims.z;
        int cols = dims.z;
        int rows = dims.x * dims.y;

        dim3 threads(256, 1, 1);
        dim3 tblocks(idiv(size, threads.x), 1, 1);
        deapodize1d_kernel <<< tblocks, threads, 0, stream >>> (arr, W, beta, cols, rows);
    }
} // namespace tomocam
