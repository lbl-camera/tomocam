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

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "dist_array.h"

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
        T *ptr_;

      public:
        DeviceArray() : ptr_(NULL) {}

        // does not check if the pointer is a valid device memory
        DeviceArray(dim3_t d, T * ptr): dims_(d), ptr_(ptr) {
            size_ = dims_.x * dims_.y * dims_.z;
        } 

        // explicit destructor
        void free() { if (ptr_) cudaFree(ptr_); }

        // set size of the window
        void set_size(size_t len) { size_ = len; }

        void set_d_array(T * arr) { ptr_ = arr; }

        void set_dims(dim3_t d) { 
            dims_ = d;
            size_ = d.x * d.y * d.z;
        }

        // at some point we'll need access to the pointer
        __host__ __device__ 
        T *d_array() { return ptr_; }

        // size of the array
        __host__ __device__
        size_t size() const { return size_; }

        // get array dims
        __host__ __device__ 
        dim3_t dims() const { return dims_; }

        // indexing 1-D
        __device__ 
        T & operator[](int i) {
            return ptr_[i];
        }

        // indexing 3-D
        __device__ 
        T & operator() (int i, int j, int k) {
            return ptr_[i * dims_.y * dims_.z + j * dims_.z + k];
        }
    };


    // convenience fuctions
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(Partition<T> p, cudaStream_t stream) {
        T * ptr = NULL;
        cudaMalloc((void **) &ptr, sizeof(T) * p.size());
        cudaMemcpyAsync(ptr, p.begin(), sizeof(T) * p.size(), cudaMemcpyHostToDevice, stream);
        DeviceArray<T> d_arr(p.dims(), ptr);
        return d_arr;
    }
   
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(dim3_t dims, T *h_ptr, cudaStream_t stream) {
        T * ptr = NULL;
        size_t size = dims.x * dims.y * dims.z;
        cudaMalloc((void **) &ptr, sizeof(T) * size);
        cudaMemcpyAsync(ptr, h_ptr, sizeof(T) * size, cudaMemcpyHostToDevice, stream);
        DeviceArray<T> d_arr(dims, ptr);
        return d_arr;
    }

    template <typename T>
    void copy_fromDeviceArray(Partition<T> dst, DeviceArray<T> src, cudaStream_t stream) {
        cudaMemcpyAsync(dst.begin(), src.d_array(), sizeof(T) * dst.size(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        src.free();
    }
} // namespace tomocam

#endif // TOMOCAM_DEV_ARRAY__H
