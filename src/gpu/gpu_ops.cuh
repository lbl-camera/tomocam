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
#    define GPU_OPS__H

namespace tomocam {
    namespace gpu_ops {
        /**************************
         * Add device arrays      *
         **************************/
        template <typename T>
        void add_arrays(const T *, const T *, T *, int, cudaStream_t);

        /**************************
         * subtract device arrays *
         **************************/
        template <typename T>
        void subtract_arrays(const T *, const T *, T *, int, cudaStream_t);

        /**************************
         * multiply device arrays *
         **************************/
        template <typename T>
        void multiply_arrays(const T *, const T *, T *, int, cudaStream_t);

        /**************************
         * broadcast and multiply device arrays *
         **************************/
        template <typename T>
        void broadcast_multiply(const T *, const T *, T *, dim3_t, cudaStream_t);

        /**************************
         * divide device arrays *
         **************************/
        template <typename T>
        void divide_arrays(const T *, const T *, T *, int, cudaStream_t);

        /******************************
         * multiply array with scalar *
         ******************************/
        template <typename T>
        void scale_array(const T *, T, T *, int, cudaStream_t);

        /**************************
         * add array and a scalar *
         **************************/
        template <typename T>
        void shift_array(const T *, T, T *, int, cudaStream_t);

        /***************************
         * initialize device array *
         ***************************/
        template <typename T>
        void init_array(T *, T, int, cudaStream_t);

        /***************************
         * dot product
         ***************************/
        float dot(const float *, const float *, int, cudaStream_t);
        
    } // namespace gpu_ops
} // namespace tomocam

#endif // GPU_OPS__H
