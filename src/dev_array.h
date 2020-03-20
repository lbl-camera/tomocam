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

#include "types.h"
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
        T *dev_ptr_;
        int2 halo_;

      public:
        DeviceArray() : dev_ptr_(NULL) {}

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

        // explicit destructor
        void free() { if (dev_ptr_) cudaFree(dev_ptr_); }

        // reset dims
        void dims(dim3_t d) { 
            dims_ = d; 
            size_ = dims_.x * dims_.y * dims_.z;
        }

        // reset dev_ptr
        void dev_ptr(T * p) { dev_ptr_ = p; }

        // at some point we'll need access to the pointer
        __host__ __device__ 
        T *dev_ptr() { return dev_ptr_; }

        // size of the array
        __host__ __device__
        size_t size() const { return size_; }

        // get array dims
        __host__ __device__ 
        dim3_t dims() const { return dims_; }

        // indexing 1-D
        __device__ 
        T & operator[](int i) {
            return dev_ptr_[i];
        }
  
        // indexing 3-D
        __device__
        T & operator[](int3 i) {
            return dev_ptr_[i.x * dims_.y * dims_.z + i.y * dims_.z + i.z];
        }

        // indexing 3-D
        __device__ 
        T & operator() (int i, int j, int k) {
            return dev_ptr_[i * dims_.y * dims_.z + j * dims_.z + k];
        }

        // indexing ...
        // -- with halo excluded
        // -- check for bounds, return 0 if outside
        __device__
        T at(int i, int j, int k) {
            int ii = i + halo_.x;
            if ((ii < 0) || (ii > dims_.x - 1)) 
                return 0;
            if ((j < 0) || (j > dims_.y - 1))
                return 0;
            if ((k < 0) || (k > dims_.z - 1))
                return 0;
            return dev_ptr_[ii * dims_.y * dims_.z + j * dims_.z + k];
        }
    };
    typedef DeviceArray<float> dev_arrayf;
    typedef DeviceArray<cuComplex_t> dev_arrayc;

    // copy elements and cast into complex types
    inline dev_arrayc DeviceArray_fromHostR2C(Partition<float> p,  cudaStream_t s) {
        cudaError_t status;
        cuComplex_t * dst = NULL;
        cudaMalloc((void **) &dst, p.size() * sizeof(cuComplex_t));
        size_t spitch = sizeof(float);
        size_t dpitch = sizeof(cuComplex_t);
        size_t width = sizeof(float);
        float * src = p.begin();
        status = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, p.size(), cudaMemcpyHostToDevice, s);
        if (status != cudaSuccess)
            throw "failed to copy-cast (R2C) to device";
        dev_arrayc d_arr(p.dims(), dst, p.halo()); 
        return d_arr;
    }

    // copy elements to host and cast complex -> float
    inline void copy_fromDeviceArrayC2R(Partition<float> p, dev_arrayc d_arr, cudaStream_t s) {
        cudaError_t status;
        size_t spitch = sizeof(cuComplex_t);
        size_t dpitch = sizeof(float);
        size_t width = sizeof(float);
        float * dst = p.begin(); 
        cuComplex_t * src = d_arr.dev_ptr();
        status = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, p.size(), cudaMemcpyDeviceToHost, s);
        if (status != cudaSuccess)
            throw "failed to copy-cast (C2R) from device";
    }
   
    // add padding on device
    inline void addPadding(dev_arrayc &d_arr, int padding, int dim, cudaStream_t s) {
        dim3_t d0 = d_arr.dims();
        dim3_t d1 = d0;
        int j_offset = 0;
        if (dim==1) {
            d1.z += 2 * padding;
        } else if (dim==2) {
            d1.y += 2 * padding;
            d1.z += 2 * padding;
            j_offset = padding;
        } else 
            throw "illegal padding dimensions";
            
        size_t bytes = d1.x * d1.y * d1.z * sizeof(cuComplex_t);
        cuComplex_t * ptr = NULL;
        cuComplex_t * src = d_arr.dev_ptr();
        cudaMalloc((void **) &ptr, bytes);
        cudaMemsetAsync(ptr, 0, bytes, s);
        for (int i = 0; i < d0.x; i++) {
            for (int j = 0; j < d0.y; j++) {
                size_t offset0 = i * d0.y * d0.z + j * d0.z;
                size_t offset1 = i * d1.y * d1.z + (j + j_offset) * d1.z + padding;
                cudaMemcpyAsync(ptr + offset1, src + offset0, d0.z * sizeof(cuComplex_t), cudaMemcpyDeviceToDevice, s);
            }
        }
        d_arr.dims(d1);
        d_arr.dev_ptr(ptr);
        cudaFree(src);
    }
     
    // strip padding 
    inline void stripPadding(dev_arrayc &d_arr, int padding, int dim, cudaStream_t s) {
        dim3_t d0 = d_arr.dims();
        dim3_t d1 = d0;
        int j_offset = 0;
        if (dim==1) {
            d1.z -= 2 * padding;
        } else if (dim==2) {
            d1.y -= 2 * padding;
            d1.z -= 2 * padding;
            j_offset = padding;
        } else 
            throw "illegal padding dimensions";
        
        size_t bytes = d1.x * d1.y * d1.z * sizeof(cuComplex_t);
        cuComplex_t * ptr = NULL;
        cuComplex_t * src = d_arr.dev_ptr();
        cudaMalloc((void **) &ptr, bytes);
        for (int i = 0; i < d1.x; i++) {
            for (int j = 0; j < d1.y; j++) {
                size_t offset0 = i * d0.y * d0.z + (j + j_offset) * d0.z + padding;
                size_t offset1 = i * d1.y * d1.z + j * d1.z;
                cudaMemcpyAsync(ptr + offset1, src + offset0, d1.z * sizeof(cuComplex_t), cudaMemcpyDeviceToDevice, s);
            }
        }
        d_arr.dims(d1);
        d_arr.dev_ptr(ptr); 
        cudaFree(src);
    }

    // create DeviceArray from Partition
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(Partition<T> p, cudaStream_t stream) {
        cudaError_t status;
        T * ptr = NULL;
        status = cudaMalloc((void **) &ptr, sizeof(T) * p.size());
        if (status != cudaSuccess) throw "failed to allocate momeory";
        
        status = cudaMemcpyAsync(ptr, p.begin(), sizeof(T) * p.size(), cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) throw "failed to copy array to device";
        DeviceArray<T> d_arr(p.dims(), ptr, p.halo());
        return d_arr;
    }
   
    // create DeviceArray from raw pointer
    template <typename T>
    DeviceArray<T> DeviceArray_fromHost(dim3_t dims, T *h_ptr, cudaStream_t stream) {
        cudaError_t status;
        T * ptr = NULL;
        size_t size = dims.x * dims.y * dims.z;
        status = cudaMalloc((void **) &ptr, sizeof(T) * size);
        if (status != cudaSuccess) throw "failed to allocate momeory";

        status = cudaMemcpyAsync(ptr, h_ptr, sizeof(T) * size, cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) throw "failed to copy array to device";

        return DeviceArray<T>(dims, ptr);
    }

    // create empty device array and set everything to zero
    template <typename T>
    DeviceArray<T> DeviceArray_fromDims(dim3_t dims, cudaStream_t stream) {
        cudaError_t status;
        T * ptr = NULL;
        size_t bytes = dims.x * dims.y * dims.z * sizeof(T);
        status = cudaMalloc((void **) &ptr, bytes);
        if (status != cudaSuccess) throw "failed to allocate momeory";

        status = cudaMemsetAsync(ptr, 0, bytes, stream);
        if (status != cudaSuccess) throw "failed to initialize memory to zeros";

        return DeviceArray<T>(dims, ptr);
    }

    // copy data from DeviceArray -> Partition
    template <typename T>
    void copy_fromDeviceArray(Partition<T> dst, DeviceArray<T> src, cudaStream_t stream) {
        cudaError_t status;
        status = cudaMemcpyAsync(dst.begin(), src.dev_ptr(), sizeof(T) * dst.size(), cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) throw "failed to copy array to host from device";
    }
} // namespace tomocam

#endif // TOMOCAM_DEV_ARRAY__H
