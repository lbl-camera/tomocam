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

#ifndef TOMOCAM_DEV_MEMORY__H
#define TOMOCAM_DEV_MEMORY__H

#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>

#include "common.h"
#include "dist_array.h"
#include "types.h"
#include "utils.cuh"

namespace tomocam {
    namespace gpu {

        /**
         * @brief DeviceMemory class is a thin wrapper around a device pointer
         * that is created as an alias DeviceArray<T> class.
         * It is implictly convertted from DeviceArray<T> to DeviceMemory<T>
         * @tparam T data type
         */
        template <typename T>
        class DeviceMemory {
          protected:
            dim3_t dims_;
            size_t size_;
            T *dev_ptr_;
            int2 halo_;

          public:
            DeviceMemory(dim3_t d, int2 h, T *ptr) :
                dims_(d), halo_(h), dev_ptr_(ptr) {
                size_ = d.x * d.y * d.z;
            }

            __host__ __device__ T *dev_ptr() { return dev_ptr_; }

            // size of the array
            __host__ __device__ size_t size() const { return size_; }

            // get array dims
            __host__ __device__ dim3_t dims() const { return dims_; }

            __device__ T &operator[](int i) { return dev_ptr_[i]; }

            // indexing 3-D
            __device__ T &operator[](int3 i) {
                return dev_ptr_[i.x * dims_.y * dims_.z + i.y * dims_.z + i.z];
            }
            __device__ const T &operator[](int3 i) const {
                return dev_ptr_[i.x * dims_.y * dims_.z + i.y * dims_.z + i.z];
            }

            // indexing 3-D
            __device__ T &operator()(int i, int j, int k) {
                return dev_ptr_[i * dims_.y * dims_.z + j * dims_.z + k];
            }
            __device__ const T &operator()(int i, int j, int k) const {
                return dev_ptr_[i * dims_.y * dims_.z + j * dims_.z + k];
            }

            // indexing ...
            // -- with halo excluded
            // -- check for bounds, return 0 if outside
            __device__ T at(int ii, int j, int k) const {
                int i = ii + halo_.x;
                if ((i < 0) || (i > dims_.x - 1)) return 0;
                if ((j < 0) || (j > dims_.y - 1)) return 0;
                if ((k < 0) || (k > dims_.z - 1)) return 0;
                return dev_ptr_[i * dims_.y * dims_.z + j * dims_.z + k];
            }
        };

        typedef DeviceMemory<float> DeviceMemoryf;
        typedef DeviceMemory<double> DeviceMemoryd;
        typedef DeviceMemory<cuda::std::complex<float>> DeviceMemorycf;
        typedef DeviceMemory<cuda::std::complex<double>> DeviceMemorycd;

    } // namespace gpu
} // namespace tomocam

#endif // TOMOCAM_DEV_MEMORY__H
