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

#include "types.h"
#include "common.h"
#include "dist_array.h"
#include "utils.cuh"

#ifdef __NVCC__
#include "dev_memory.cuh"
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

        // Allocate space 
        DeviceArray(dim3_t d): dims_(d) {
            halo_ = {0, 0};
            size_ = d.x * d.y * d.z;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
        }

        // Allocate space with halo
        DeviceArray(dim3_t d, int *h): dims_(d) {
            halo_ = {h[0], h[1]};
            size_ = (d.x + h[0] + h[1]) * d.y * d.z;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
        }

        // does not check if the pointer is a valid device memory
        DeviceArray(dim3_t d, T *ptr): dims_(d), dev_ptr_(ptr) {
            halo_ = {0, 0};
            size_ = dims_.x * dims_.y * dims_.z;
        } 

        // create with halo
        DeviceArray(dim3_t d, T *ptr, int *h): dims_(d), dev_ptr_(ptr) {
            halo_.x = h[0];
            halo_.y = h[1];
            size_ = dims_.x * dims_.y * dims_.z;
        }

        // destructor
        ~DeviceArray() { if (dev_ptr_) SAFE_CALL(cudaFree(dev_ptr_)); }

        //  copy constructor
        DeviceArray(const DeviceArray<T> &rhs) {
            dims_ = rhs.dims_;
            size_ = rhs.size_;
            halo_ = rhs.halo_;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
            SAFE_CALL(cudaMemcpy(dev_ptr_, rhs.dev_ptr_, sizeof(T) * size_, cudaMemcpyDeviceToDevice));
        }

        // assignment operator
        DeviceArray<T> operator=(const DeviceArray & rhs) {
            dims_ = rhs.dims_;
            size_ = rhs.size_;
            halo_ = rhs.halo_;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
            SAFE_CALL(cudaMemcpy(dev_ptr_, rhs.dev_ptr_, sizeof(T) * size_, cudaMemcpyDeviceToDevice));
            return *this;
        }

        #ifdef __NVCC__
        // convert to DeviceMemory
        operator  DeviceMemory<T>() {
            return DeviceMemory<T>(dims_, halo_, dev_ptr_);
        }
        #endif
                
        // access to the devie-pointer
        T *dev_ptr() { return dev_ptr_; }

        // size of the array
        size_t size() const { return size_; }

        // get array dims
        dim3_t dims() const { return dims_; }

        // dump to a binary file
        void tofile(const char *filename) {
            T * h_ptr = new T[size_];
            SAFE_CALL(cudaMemcpy(h_ptr, dev_ptr_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
            std::ofstream out(filename, std::ios::binary);
            out.write((char *) h_ptr, sizeof(T) * size_);
            out.close();
            delete [] h_ptr;
        }
    };
    typedef DeviceArray<int> dev_arrayI;
    typedef DeviceArray<float> dev_arrayF;
    typedef DeviceArray<cuComplex_t> dev_arrayZ;

    // create DeviceArray from Partition
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(Partition<T> p, cudaStream_t stream) {
        T * ptr = NULL;
        SAFE_CALL(cudaMalloc((void **) &ptr, sizeof(T) * p.size()));
        SAFE_CALL(cudaMemcpyAsync(ptr, p.begin(), sizeof(T) * p.size(), cudaMemcpyHostToDevice, stream));
        DeviceArray<T> d_arr(p.dims(), ptr, p.halo());
        return d_arr;
    }
   
    // create DeviceArray from raw pointer
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(dim3_t dims, T *h_ptr, cudaStream_t stream) {
        T * ptr = NULL;
        size_t size = dims.x * dims.y * dims.z;
        SAFE_CALL(cudaMalloc((void **) &ptr, sizeof(T) * size));
        SAFE_CALL(cudaMemcpyAsync(ptr, h_ptr, sizeof(T) * size, cudaMemcpyHostToDevice, stream));
        return DeviceArray<T>(dims, ptr);
    }

    // create empty device array and set everything to zero
    template <typename T>
    DeviceArray<T> DeviceArray_fromDims(dim3_t dims, cudaStream_t stream) {
        T * ptr = NULL;
        size_t bytes = dims.x * dims.y * dims.z * sizeof(T);
        SAFE_CALL(cudaMalloc((void **) &ptr, bytes));
        SAFE_CALL(cudaMemsetAsync(ptr, 0, bytes, stream));
        return DeviceArray<T>(dims, ptr);
    }

    // copy data from DeviceArray -> Partition
    template <typename T>
    void copy_fromDeviceArray(Partition<T> dst, DeviceArray<T> src, cudaStream_t stream) {
        SAFE_CALL(cudaMemcpyAsync(dst.begin(), src.dev_ptr(), sizeof(T) * dst.size(), cudaMemcpyDeviceToHost, stream));
    }

} // namespace tomocam

#endif // TOMOCAM_DEV_ARRAY__H
