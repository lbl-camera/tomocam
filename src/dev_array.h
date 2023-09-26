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

#ifndef TOMOCAM_DEV_ARRAY__H
#define TOMOCAM_DEV_ARRAY__H

#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "dist_array.h"
#include "types.h"

#include "utils.cuh"

#ifdef __NVCC__
#    include "dev_memory.cuh"
#endif // __NVCC__

namespace tomocam {

    template <typename T>
    class DeviceArray {
        /*
         * Making shallow copies by design.
         * The objects are designed to be initialized from python.
         * once created, the ownership is passed to the python, which
         * is responsible for calling the destructor.
         */
      protected:
        dim3_t dims_;
        size_t size_;
        T *dev_ptr_;
        int2 halo_;

      public:
        DeviceArray() = default;

        // DeviceArray from dims
        DeviceArray(dim3_t dims): dims_(dims) {
            halo_ = {0, 0};
            size_ = dims_.x * dims_.y * dims_.z;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
        }

        // create with halo
        DeviceArray(dim3_t d, int *h) : dims_(d) {
            halo_.x = h[0];
            halo_.y = h[1];
            size_ = dims_.x * dims_.y * dims_.z;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
        }

        // copy constructor
        DeviceArray(const DeviceArray<T> &rhs) {
            halo_ = rhs.halo_;
            dims_ = rhs.dims_;
            size_ = rhs.size_;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
            SAFE_CALL(cudaMemcpy(dev_ptr_, rhs.dev_ptr_,
                sizeof(T) * size_, cudaMemcpyDeviceToDevice));
        }

        DeviceArray<T> operator=(const DeviceArray<T> &rhs) {
            halo_ = rhs.halo_;
            dims_ = rhs.dims_;
            size_ = rhs.size_;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
            SAFE_CALL(cudaMemcpy(dev_ptr_, rhs.dev_ptr_,
                sizeof(T) * size_, cudaMemcpyDeviceToDevice));
            return *this;
        }

        // explicit destructor
        ~DeviceArray() {
            if (dev_ptr_) SAFE_CALL(cudaFree(dev_ptr_));
        }

        // at some point we'll need access to the pointer
        T *dev_ptr() { return dev_ptr_; }

        // const ptr
        const T *dev_ptr() const { return dev_ptr_; }

        // size of the array
        size_t size() const { return size_; }
        size_t bytes() const { return sizeof(T) * size_; }
        dim3_t dims() const { return dims_; }

#ifdef __NVCC__
        // convert to DeviceMemory
        operator DeviceMemory<T>() {
            return DeviceMemory<T>(dims_, halo_, dev_ptr_);
        }

        operator DeviceMemory<T>() const {
            return DeviceMemory<T>(dims_, halo_, dev_ptr_);
        }
#endif
    };
    typedef DeviceArray<float> dev_arrayf;
    typedef DeviceArray<cuComplex_t> dev_arrayc;

    // create DeviceArray from Partition
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(Partition<T> p, cudaStream_t stream) {
        DeviceArray<T> d_arr(p.dims());
        size_t bytes = sizeof(T) * p.size();
        SAFE_CALL(cudaMemcpyAsync(
            d_arr.dev_ptr(), p.begin(), bytes, cudaMemcpyHostToDevice, stream));
        return d_arr;
    }

    // create DeviceArray from raw pointer
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(dim3_t dims, T *ptr, cudaStream_t stream) {
        DeviceArray<T> d_arr(dims);
        size_t bytes = sizeof(T) * dims.x * dims.y * dims.z;
        SAFE_CALL(cudaMemcpyAsync(
            d_arr.dev_ptr(), ptr, bytes, cudaMemcpyHostToDevice, stream));
        return d_arr;
    }

    // create empty device array and set everything to zero
    template <typename T>
    DeviceArray<T> DeviceArray_fromDims(dim3_t dims, cudaStream_t stream) {
        DeviceArray<T> d_arr(dims);
        SAFE_CALL(cudaMemsetAsync(d_arr.dev_ptr(), 0, d_arr.bytes(), stream));
        return d_arr;
    }

    // copy data from DeviceArray -> Partition
    template <typename T>
    void copy_fromDeviceArray(
        Partition<T> dst, DeviceArray<T> src, cudaStream_t stream) {
        SAFE_CALL(cudaMemcpyAsync(dst.begin(), src.dev_ptr(),
            sizeof(T) * dst.size(), cudaMemcpyDeviceToHost, stream));
    }
} // namespace tomocam

#endif // TOMOCAM_DEV_ARRAY__H
