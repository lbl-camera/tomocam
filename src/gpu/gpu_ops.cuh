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

#include "utils.cuh"

#ifndef GPU_OPS__H
#define GPU_OPS__H

namespace tomocam {
    namespace gpu {

        /** Add device arrays 
         * @param[in] a - first array
         * @param[in] b - second array
         * @param[out] c - output array
         * @param[in] n - number of elements in the array
         * @param[in] stream - cuda stream
         */
        template <typename T>
        void add_arrays(const T *, const T *, T *, int, cudaStream_t);

        /** subtract device arrays 
         * @param[in] a - first array
         * @param[in] b - second array
         * @param[out] c - output array
         * @param[in] n - number of elements in the array
         * @param[in] stream - cuda stream
         */ 
        template <typename T>
        void subtract_arrays(const T *, const T *, T *, int, cudaStream_t);

         /** multiply device arrays.
           * @param[in] a - first array
           * @param[in] b - second array
           * @param[out] c - output array
           * @param[in] n - number of elements in the array
           * @param[in] stream - cuda stream
           */
        template <typename T>
        void multiply_arrays(const T *, const T *, T *, int, cudaStream_t);

        /** broadcast and multiply device arrays *
          * @param[in] a - first array
          * @param[in] b - second array
          * @param[out] c - output array
          * @param[in] n - number of elements in the array
          * @param[in] stream - cuda stream
          */
        template <typename T>
        void broadcast_multiply(const T *, const T *, T *, dim3_t, cudaStream_t);

        /** divide device arrays
          * @param[in] a - first array
          * @param[in] b - second array
          * @param[out] c - output array
          * @param[in] n - number of elements in the array
          * @param[in] stream - cuda stream
          */
        template <typename T>
        void divide_arrays(const T *, const T *, T *, int, cudaStream_t);

        /** multiply array with scalar
          * @param[in] a - input array
          * @param[in] scalar - scalar
          * @param[out] c - output array
          * @param[in] n - number of elements in the array
          * @param[in] stream - cuda stream
          */
        template <typename T>
        void scale_array(const T *, T, T *, int, cudaStream_t);

        /** add array and a scalar 
          * @param[in] a - input array
          * @param[in] scalar - scalar
          * @param[out] c - output array
          * @param[in] n - number of elements in the array
          * @param[in] stream - cuda stream
          */
        template <typename T>
        void shift_array(const T *, T, T *, int, cudaStream_t);

        /** initialize device array *
          * @param[in] a - input array
          * @param[in] scalar - scalar
          * @param[in] n - number of elements in the array
          * @param[in] stream - cuda stream
          */       
        template <typename T>
        void init_array(T *, T, int, cudaStream_t);

        /* dot product of two arrays
         * @param[in] a - first array
         * @param[in] b - second array
         * @param[in] n - number of elements in the array
         * @param[in] stream - cuda stream
         * @return dot product
         */
        template <typename T>
        T dot(const T *, const T *, int, cudaStream_t);

        /** cast array from real to complex
         * @param[in] a - input array
         * @param[out] b - output array
         * @param[in] n - number of elements in the array
         * @param[in] stream - cuda stream
         */
        template <typename T>
        void cast_array_to_complex(
            const T *, gpu::complex_t<T> *, int, cudaStream_t);

        /** cast array from complex to real
         * @param[in] a - input array
         * @param[out] b - output array
         * @param[in] n - number of elements in the array
         * @param[in] stream - cuda stream
         */
        template <typename T>
        void cast_array_to_real(
            const gpu::complex_t<T> *, T *, int, cudaStream_t);

        /** Given R and theta, compute points on the polar grid
         * @param[in] num_rows - number of rows
         * @param[in] num_proj - number of projections angles
         * @param[out] x - x coordinate
         * @param[out] y - y coordinate
         * @param[in] theta - projection angles
         * @param[in] stream - cuda stream
         */
        template <typename T>
        void make_nugrid(int, int, T *, T *, const T *, cudaStream_t);

    } // namespace gpu
} // namespace tomocam

#endif // GPU_OPS__H
