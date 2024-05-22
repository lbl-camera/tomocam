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

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#include "common.h"
#include "types.h"

#ifndef TOMOCAM_UTILS__CUH
#define TOMOCAM_UTILS__CUH

#define SAFE_CALL(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
            cudaGetErrorString(code),
            file,
            line);
        if (abort) exit(code);
    }
}

#define __deviceI__ __forceinline__ __device__
#define __devhstI__ __forceinline__ __device__ __host__

namespace tomocam {
    namespace gpu {

        template <typename T>
        __deviceI__ T *SharedMemory() {
            extern __shared__ unsigned char __smem[];
            return reinterpret_cast<T *>(__smem);
        }

        inline unsigned int idiv(size_t a, int b) {
            if (a % b) return (a / b + 1);
            else return (a / b);
        }

        // cuda doesn't like when any thread block has:
        // ...  x = 1 and y >, z > 1, or
        // ...  x = 1, y = 1 and z > 1
        struct Grid {
            dim3 threads_;
            dim3 blocks_;

            Grid(size_t nx) {
                threads_ = {256, 1, 1};
                blocks_ = {idiv(nx, threads_.x), 1, 1};
            }

            Grid(int3 d) {
                threads_ = {16, 16, 1};
                blocks_ = {idiv(d.z, threads_.x),
                    idiv(d.y, threads_.y),
                    idiv(d.x, threads_.z)};
            }

            Grid(dim3_t d) {
                threads_ = {16, 16, 1};
                blocks_ = {idiv(d.z, threads_.x),
                    idiv(d.y, threads_.y),
                    idiv(d.x, threads_.z)};
            }

            dim3 blocks() { return blocks_; }
            dim3 threads() { return threads_; }
        };

// calculate thread global index
#    ifdef __NVCC__
        __deviceI__ int Index1D() {
            return (blockDim.x * blockIdx.x + threadIdx.x);
        }

        __deviceI__ int3 Index3D() {
            int3 idx;
            idx.x = blockDim.z * blockIdx.z + threadIdx.z;
            idx.y = blockDim.y * blockIdx.y + threadIdx.y;
            idx.z = blockDim.x * blockIdx.x + threadIdx.x;
            return idx;
        }
#    endif // __NVCC__

        // check if indices are in range
        __deviceI__ bool operator<(int3 i, dim3_t d) {
            if ((i.x < d.x) && (i.y < d.y) && (i.z < d.z)) return true;
            else
                return false;
        }

        __devhstI__ bool operator<(int3 a, int3 b) {
            if ((a.x < b.x) && (a.y < b.y) && (a.z < b.z)) return true;
            else
                return false;
        }

        __devhstI__ int3 operator+(int3 a, int3 b) {
            return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
        }

        __devhstI__ int3 operator-(int3 a, int3 b) {
            return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
        }

        __devhstI__ int3 operator/(int3 a, int b) {
            return make_int3(a.x / b, a.y / b, a.z / b);
        }

    } // namespace gpu
} // namespace tomocam
#endif // TOMOCAM_UTILS__CUH
