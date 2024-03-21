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
#include <iostream>

#include <cuda.h>
#include "utils.cuh"
#include "gpu_ops.cuh"

namespace tomocam {
    namespace gpu_ops {
        /**************************
         * Add device arrays      *
         **************************/
        template <typename T>
        __global__ void gpu_add_arrays(const T *a, const T *b, T *result, int size) {
            int idx = Index1D();
            if (idx < size) result[idx] = a[idx] + b[idx];
        }

        template <typename T>
        void add_arrays(const T *a, const T *b, T *result, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_add_arrays<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, result, size);
        }

        // specialize
        template void add_arrays(const float *, const float *, float *, int, cudaStream_t);
        template void add_arrays(const cuComplex_t *, const cuComplex_t *, cuComplex_t *, int, cudaStream_t);

        /**************************
         * subtract device arrays *
         **************************/
        template <typename T>
        __global__ void gpu_subtract_arrays(const T *a, const T *b, T *result, int size) {
            int idx = Index1D();
            if (idx < size) result[idx] = a[idx] - b[idx];
        }

        template <typename T>
        void subtract_arrays(const T *a, const T *b, T *result, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_subtract_arrays<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, result, size);
        }

        // specialize
        template void subtract_arrays(const float *, const float *, float *, int, cudaStream_t);
        template void subtract_arrays(const cuComplex_t *, const cuComplex_t *, cuComplex_t *, int, cudaStream_t);

        /**************************
         * multiply device arrays *
         **************************/
        template <typename T>
        __global__ void gpu_multiply_arrays(const T *a, const T *b, T *result, int size) {
            int idx = Index1D();
            if (idx < size) result[idx] = a[idx] * b[idx];
        }

        template <typename T>
        void multiply_arrays(const T *a, const T *b, T *result, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_multiply_arrays<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, result, size);
        }

        // specialize
        template void multiply_arrays(const float *, const float *, float *, int, cudaStream_t);
        template void multiply_arrays(const cuComplex_t *, const cuComplex_t *, cuComplex_t *, int, cudaStream_t);


        /*************************
         * broadcast and multiply
         ************************/
        template <typename T>
        __global__ void gpu_broadcast_multiply(const T *a, const T *b, T *result, int3 dims) {
            int3 idx = Index3D();
            if (idx < dims) {
                int i0 = idx.x * dims.y * dims.z + idx.y * dims.z + idx.z;
                int i1 = idx.y * dims.z + idx.z;
                result[i0] = a[i0] * b[i1]; 
            }
        }

        template <typename T>
        void broadcast_multiply(const T *a, const T *b, T *c, dim3_t d, cudaStream_t stream) {
            int3 dims = to_int3(d);
            Grid grid(dims);
            gpu_broadcast_multiply<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, c, dims);
        } 
        template void broadcast_multiply(const float *, const float *, float *, dim3_t, cudaStream_t);
        template void broadcast_multiply(const cuComplex_t *, const cuComplex_t *, cuComplex_t *, dim3_t, cudaStream_t);

        /**************************
         * divide device arrays *
         **************************/
        template <typename T>
        __global__ void gpu_divide_arrays(const T *a, const T *b, T *result, int size) {
            int idx = Index1D();
            if (idx < size) result[idx] = a[idx] / b[idx];
        }

        template <typename T>
        void divide_arrays(const T *a, const T *b, T *result, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_divide_arrays<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, result, size);
        }

        // specialize
        template void divide_arrays(const float *, const float *, float *, int, cudaStream_t);
        template void divide_arrays(const cuComplex_t *, const cuComplex_t *, cuComplex_t *, int, cudaStream_t);

        /******************************
         * multiply array with scalar *
         ******************************/
        template <typename T>
        __global__ void gpu_scale_array(const T *a, T b, T *result, int size) {
            int idx = Index1D();
            if (idx < size) result[idx] = a[idx] * b;
        }

        template <typename T>
        void scale_array(const T *a, T b, T *result, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_scale_array<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, result, size);
        }

        // specialize
        template void scale_array(const float *, float, float *, int, cudaStream_t);
        template void scale_array(const cuComplex_t *, cuComplex_t, cuComplex_t *, int, cudaStream_t);

        /**************************
         * add array and a scalar *
         **************************/
        template <typename T>
        __global__ void gpu_shift_array(const T *a, T b, T *result, int size) {
            int idx = Index1D();
            if (idx < size) result[idx] = a[idx] + b;
        }

        template <typename T>
        void shift_array(const T *a, T b, T *result, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_shift_array<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, result, size);
        }

        // specialize
        template void shift_array(const float *, float, float *, int, cudaStream_t);
        template void shift_array(const cuComplex_t *, cuComplex_t, cuComplex_t *, int, cudaStream_t);

        /***************************
         * initialize device array *
         ***************************/
        template <typename T>
        __global__ void gpu_init_array(T *a, T b, int size) {
            int idx = Index1D();
            if (idx < size) a[idx] = b;
        }

        template <typename T>
        void init_array(T *a, T b, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_init_array<T><<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, size);
        }

        // specialize
        template void init_array(float *, float, int, cudaStream_t);
        template void init_array(cuComplex_t *, cuComplex_t, int, cudaStream_t);


        /***************************
         * dot product
         ***************************/
        __global__ void
        gpu_dot(const float *a, const float *b, float *c, int size) {

            int idx = Index1D();
            int tid = threadIdx.x;
            extern __shared__ float temp[];

            if (idx < size) {
                temp[tid] = a[idx] * b[idx];
                __syncthreads();

                // reduce
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) temp[tid] += temp[tid + s];
                    __syncthreads();
                }
                if (tid == 0) atomicAdd(c, temp[tid]);
            }
        }

        float dot(const float *a,  const float *b, int size, cudaStream_t stream) {

            Grid grid(size);
            float result;
            float *d_result; 

            SAFE_CALL(cudaMalloc(&d_result, sizeof(float)));
            SAFE_CALL(cudaMemsetAsync(d_result, 0, sizeof(float), stream));
            size_t shamem = grid.threads().x * sizeof(float);
            gpu_dot<<<grid.blocks(), grid.threads(), shamem, stream>>>(a, b, d_result, size);
            SAFE_CALL(cudaMemcpyAsync(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream));
            return result;
        }

    } // namespace gpu_ops
} // namespace tomocam
