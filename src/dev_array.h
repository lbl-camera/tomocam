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

#include "common.h"
#include "dist_array.h"
#include "gpu/gpu_ops.cuh"
#include "gpu/utils.cuh"
#include "partition.h"
#include "types.h"
#include <ctime>
#include <cuda.h>

#ifdef __NVCC__
#include "gpu/dev_memory.cuh"
#endif // __NVCC__

namespace tomocam {

    template <typename T>
    class DeviceArray {
      protected:
        dim3_t dims_;
        size_t size_;
        T *dev_ptr_;
        int2 halo_;

      public:
        DeviceArray() = default;

        // Allocate space
        DeviceArray(dim3_t d) : dims_(d) {
            halo_ = {0, 0};
            size_ = d.x * d.y * d.z;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
        }

        // Allocate space with halo
        DeviceArray(dim3_t d, int *h) : dims_(d) {
            halo_ = {h[0], h[1]};
            size_ = d.x * d.y * d.z;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
        }

        /* create device array from partition */
        DeviceArray(const Partition<T> &rhs) {
            dims_ = rhs.dims();
            size_ = rhs.size();
            halo_ = {rhs.halo()[0], rhs.halo()[1]};
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
            SAFE_CALL(cudaMemcpy(dev_ptr_, rhs.begin(), rhs.bytes(),
                cudaMemcpyHostToDevice));
        }

        // destructor
        ~DeviceArray() {
            if (dev_ptr_) SAFE_CALL(cudaFree(dev_ptr_));
        }

        //  copy constructor
        DeviceArray(const DeviceArray<T> &rhs) {
            dims_ = rhs.dims_;
            halo_ = rhs.halo_;
            size_ = rhs.size_;
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
            size_t bytes = sizeof(T) * size_;
            SAFE_CALL(cudaMemcpy(dev_ptr_, rhs.dev_ptr_, bytes, cudaMemcpyDeviceToDevice));
        }

        // assignment operator
        DeviceArray<T> operator=(const DeviceArray &rhs) {
            dims_ = rhs.dims_;
            size_ = rhs.size_;
            halo_ = rhs.halo_;
            if (dev_ptr_) SAFE_CALL(cudaFree(dev_ptr_));
            SAFE_CALL(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
            size_t bytes = sizeof(T) * size_;
            SAFE_CALL(cudaMemcpy(dev_ptr_, rhs.dev_ptr_, bytes, cudaMemcpyDeviceToDevice));
            return *this;
        }

        // move constructor
        DeviceArray(DeviceArray<T> &&rhs) {
            dims_ = rhs.dims_;
            halo_ = rhs.halo_;
            size_ = rhs.size_;
            dev_ptr_ = std::move(rhs.dev_ptr_);
            rhs.dev_ptr_ = nullptr;
        }


        // move assignment operator
        DeviceArray<T> operator=(DeviceArray<T> &&rhs) {
            dims_ = rhs.dims_;
            size_ = rhs.size_;
            halo_ = rhs.halo_;
            dev_ptr_ = std::move(rhs.dev_ptr_);
            rhs.dev_ptr_ = nullptr;
            return *this;
        }

#ifdef __NVCC__
        // convert to DeviceMemory
        operator gpu::DeviceMemory<T>() {
            return gpu::DeviceMemory<T>(dims_, halo_, dev_ptr_);
        }

        operator gpu::DeviceMemory<T>() const {
            return gpu::DeviceMemory<T>(dims_, halo_, dev_ptr_);
        }
#endif

        // access to the device-pointer
        T *dev_ptr() { return dev_ptr_; }
        T *data() { return dev_ptr_; }

        // access to the device-pointer
        const T *dev_ptr() const { return dev_ptr_; };
        const T *data() const { return dev_ptr_; }

        // size of the array
        size_t size() const { return size_; }

        // bytes of the array
        size_t bytes() const { return sizeof(T) * size_; }

        // get array dims
        dim3_t dims() const { return dims_; }

        // get number of slices
        int nslices() const { return dims_.x; }

        // get number of rows
        int nrows() const { return dims_.y; }

        // get number of columns
        int ncols() const { return dims_.z; }

        // initialize
        void init(T v, cudaStream_t s) {
            gpu::init_array<T>(dev_ptr_, v, size_, s);
        }

        // copy to partition
        void copy_to(Partition<T> &rhs, cudaStream_t s) const {
            if (dims_ == rhs.dims()) {
                SAFE_CALL(cudaMemcpyAsync(rhs.begin(),
                    dev_ptr_,
                    sizeof(T) * size_,
                    cudaMemcpyDeviceToHost, s));
            } else {
                throw std::runtime_error("Partition dimensions do not match");
            }
        }

        /* arithmatic operations should be asychronized */

        // multiply (for FFT Convolution)
        DeviceArray<T> multiply(const DeviceArray<T> &arr, cudaStream_t s) const {
            DeviceArray<T> res(dims_);
            if (dims_ == arr.dims_)
                gpu::multiply_arrays<T>(
                    dev_ptr_, arr.dev_ptr_, res.dev_ptr_, size_, s);
            else if ((dims_.y == arr.dims_.y) && (dims_.z == arr.dims_.z)) {
                gpu::broadcast_multiply<T>(
                    dev_ptr_, arr.dev_ptr_, res.dev_ptr_, dims_, s);
            }
            return res;
        }

        // division (for normalization)
        DeviceArray<T> divide(T val, cudaStream_t s) const {
            DeviceArray<T> res(dims_);
            T val_inv = static_cast<T>(1) / val;
            gpu::scale_array<T>(dev_ptr_, val_inv, res.dev_ptr_, size_, s);
            return res;
        }

        // addition
        DeviceArray<T> add(const DeviceArray<T> &arr, cudaStream_t s) const {
            DeviceArray<T> res(dims_);
            if (dims_ == arr.dims_)
                gpu::add_arrays<T>(
                    dev_ptr_, arr.dev_ptr_, res.dev_ptr_, size_, s);
            else
                throw std::runtime_error("Array dimensions do not match");
            return res;
        }

        // subtraction
        DeviceArray<T> subtract(const DeviceArray<T> &arr, cudaStream_t s) const {
            DeviceArray<T> res(dims_);
            if (dims_ == arr.dims_)
                gpu::subtract_arrays<T>(
                    dev_ptr_, arr.dev_ptr_, res.dev_ptr_, size_, s);
            else
                throw std::runtime_error("Array dimensions do not match");
            return res;
        }

        // norm2
        T norm2(cudaStream_t s) const {
            return gpu::dot<T>(dev_ptr_, dev_ptr_, size_, s);
        }

        // dot product
        T dot(const DeviceArray<T> &arr, cudaStream_t s) const {
            return gpu::dot(dev_ptr_, arr.dev_ptr_, size_, s);
        }

        // dump to a binary file
        void tofile(const char *filename) {
            T *h_ptr = new T[size_];
            SAFE_CALL(cudaMemcpy(
                h_ptr, dev_ptr_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
            std::ofstream out(filename, std::ios::binary);
            out.write((char *)h_ptr, sizeof(T) * size_);
            out.close();
            delete[] h_ptr;
        }
    };

    typedef DeviceArray<float> DeviceArrayf;
    typedef DeviceArray<double> DeviceArrayd;
    typedef DeviceArray<gpu::complex_t<float>> DeviceArraycf;
    typedef DeviceArray<gpu::complex_t<double>> DeviceArraycd;

    // conversion from complex to real
    template <typename T>
    DeviceArray<T> real(
        const DeviceArray<gpu::complex_t<T>> &arr, cudaStream_t s) {
        DeviceArray<T> res(arr.dims());
        gpu::cast_array_to_real<T>(arr.dev_ptr(), res.dev_ptr(), arr.size(), s);
        return res;
    }

    // conversion from real to complex
    template <typename T>
    DeviceArray<gpu::complex_t<T>> complex(
        const DeviceArray<T> &arr, cudaStream_t s) {
        DeviceArray<gpu::complex_t<T>> res(arr.dims());
        gpu::cast_array_to_complex<T>(
            arr.dev_ptr(), res.dev_ptr(), arr.size(), s);
        return res;
    }

} // namespace tomocam

#endif // TOMOCAM_DEV_ARRAY__H
