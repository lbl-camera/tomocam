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
    namespace gpu {
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
        template void add_arrays(const gpu::complex_t<float> *,
            const gpu::complex_t<float> *,
            gpu::complex_t<float> *,
            int,
            cudaStream_t);
        template void add_arrays(const double *, const double *, double *, int, cudaStream_t);
        template void add_arrays(const gpu::complex_t<double> *,
            const gpu::complex_t<double> *,
            gpu::complex_t<double> *,
            int,
            cudaStream_t);

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
        template void subtract_arrays(const gpu::complex_t<float> *,
            const gpu::complex_t<float> *,
            gpu::complex_t<float> *,
            int,
            cudaStream_t);
        template void subtract_arrays(const double *, const double *, double *, int, cudaStream_t);
        template void subtract_arrays(const gpu::complex_t<double> *,
            const gpu::complex_t<double> *,
            gpu::complex_t<double> *,
            int,
            cudaStream_t);

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
        template void multiply_arrays(const gpu::complex_t<float> *,
            const gpu::complex_t<float> *,
            gpu::complex_t<float> *,
            int,
            cudaStream_t);
        template void multiply_arrays(const double *, const double *, double *, int, cudaStream_t);
        template void multiply_arrays(const gpu::complex_t<double> *,
            const gpu::complex_t<double> *,
            gpu::complex_t<double> *,
            int,
            cudaStream_t);

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
        void broadcast_multiply(const T *a, const T *b, T *c, dim3_t dims, cudaStream_t stream) {
            Grid grid(dims);
            gpu_broadcast_multiply<<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, c, dims);
        } 
        template void broadcast_multiply(const float *, const float *, float *, dim3_t, cudaStream_t);
        template void broadcast_multiply(const gpu::complex_t<float> *,
            const gpu::complex_t<float> *,
            gpu::complex_t<float> *,
            dim3_t,
            cudaStream_t);
        template void broadcast_multiply(const double *, const double *, double *, dim3_t, cudaStream_t);
        template void broadcast_multiply(const gpu::complex_t<double> *,
            const gpu::complex_t<double> *,
            gpu::complex_t<double> *,
            dim3_t,
            cudaStream_t);

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
        template void divide_arrays(const gpu::complex_t<float> *,
            const gpu::complex_t<float> *,
            gpu::complex_t<float> *,
            int,
            cudaStream_t);
        template void divide_arrays(const double *, const double *, double *, int, cudaStream_t);
        template void divide_arrays(const gpu::complex_t<double> *,
            const gpu::complex_t<double> *,
            gpu::complex_t<double> *,
            int,
            cudaStream_t);

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
        template void scale_array(const gpu::complex_t<float> *,
            gpu::complex_t<float>,
            gpu::complex_t<float> *,
            int,
            cudaStream_t);
        template void scale_array(const double *, double, double *, int, cudaStream_t);
        template void scale_array(const gpu::complex_t<double> *,
            gpu::complex_t<double>,
            gpu::complex_t<double> *,
            int,
            cudaStream_t);

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
        template void shift_array(const gpu::complex_t<float> *,
            gpu::complex_t<float>,
            gpu::complex_t<float> *,
            int,
            cudaStream_t);
        template void shift_array(const double *, double, double *, int, cudaStream_t);
        template void shift_array(const gpu::complex_t<double> *,
            gpu::complex_t<double>,
            gpu::complex_t<double> *,
            int,
            cudaStream_t);

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
        template void init_array(
            gpu::complex_t<float> *, gpu::complex_t<float>, int, cudaStream_t);
        template void init_array(double *, double, int, cudaStream_t);
        template void init_array(gpu::complex_t<double> *,
            gpu::complex_t<double>,
            int,
            cudaStream_t);

        /***************************
         * dot product
         ***************************/
        template <typename T>
        __global__ void
        gpu_dot(const T *a, const T *b, T *c, int size) {

            int idx = Index1D();
            int tid = threadIdx.x;
            T * temp = SharedMemory<T>();

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

        template <typename T>
        T dot(const T *a,  const T *b, int size, cudaStream_t stream) {

            Grid grid(size);
            T result;
            T *d_result; 

            SAFE_CALL(cudaMalloc(&d_result, sizeof(T)));
            SAFE_CALL(cudaMemsetAsync(d_result, 0, sizeof(T), stream));
            size_t shamem = grid.threads().x * sizeof(T);
            gpu_dot<<<grid.blocks(), grid.threads(), shamem, stream>>>(a, b, d_result, size);
            SAFE_CALL(cudaMemcpyAsync(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost, stream));
            return result;
        }
        // explicit instantiation
        template float dot(const float *, const float *, int, cudaStream_t);
        template double dot(const double *, const double *, int, cudaStream_t);

        /***************************
         * cast  real to complex
         ***************************/
        template <typename T>
        __global__ void gpu_cast_array_to_complex(
            const T *a, gpu::complex_t<T> *b, int size) {
            int idx = Index1D();
            if (idx < size) b[idx] = gpu::complex_t<T>(a[idx], 0);
        }

        template <typename T>
        void cast_array_to_complex(
            const T *a, gpu::complex_t<T> *b, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_cast_array_to_complex<T>
                <<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, size);
        }

        // specialize
        template void cast_array_to_complex(
            const float *, gpu::complex_t<float> *, int, cudaStream_t);
        template void cast_array_to_complex(
            const double *, gpu::complex_t<double> *, int, cudaStream_t);

        /***************************
         * cast  complex to real
         ***************************/
        template <typename T>
        __global__ void gpu_cast_array_to_real(
            const gpu::complex_t<T> *a, T *b, int size) {
            int idx = Index1D();
            if (idx < size) b[idx] = a[idx].real();
        }

        template <typename T>
        void cast_array_to_real(
            const gpu::complex_t<T> *a, T *b, int size, cudaStream_t stream) {
            Grid grid(size);
            gpu_cast_array_to_real<T>
                <<<grid.blocks(), grid.threads(), 0, stream>>>(a, b, size);
        }
        // specialize
        template void cast_array_to_real(
            const gpu::complex_t<float> *, float *, int, cudaStream_t);
        template void cast_array_to_real(
            const gpu::complex_t<double> *, double *, int, cudaStream_t);

    } // namespace gpu
} // namespace tomocam
