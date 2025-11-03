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
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "dist_array.h"
#include "gpu/gpu_ops.cuh"
#include "gpu/utils.cuh"
#include "partition.h"
#include "types.h"
#include <ctime>

#ifdef __NVCC__
#include "gpu/dev_memory.cuh"
#endif // __NVCC__

namespace tomocam {

    namespace gpuMem {

        static constexpr size_t TWOGB = 1LL << 31; 
        struct cudaDeleter {
            void operator()(void *ptr) const {
                if (ptr)
                    SAFE_CALL(cudaFree(ptr));
            }
        };

        template <class T> using cuniquePtr = std::unique_ptr<T, cudaDeleter>;

        template <class T> cuniquePtr<T> make_cuniquePtr(size_t num_elems) {
            if (num_elems > TWOGB / sizeof(T)) {
                throw std::runtime_error("over the top allocation");
            }
            T *ptr = nullptr;
            if (num_elems > 0) {
                SAFE_CALL(cudaMalloc((void **)&ptr, sizeof(T) * num_elems));
            }
            return cuniquePtr<T>(ptr);
        }

    } // namespace gpuMem

    template <typename T> class DeviceArray {
      protected:
        dim3_t dims_;
        size_t size_;
        gpuMem::cuniquePtr<T> dev_ptr_;
        int2 halo_;

      public:
        DeviceArray() : dims_({0, 0, 0}), size_(0), dev_ptr_(nullptr) {}

        // Allocate space
        DeviceArray(dim3_t d) : dims_(d) {
            halo_ = {0, 0};
            size_ = d.x * d.y * d.z;
            dev_ptr_ = gpuMem::make_cuniquePtr<T>(size_);
        }

        // Allocate space with halo
        DeviceArray(dim3_t d, int *h) : dims_(d) {
            halo_ = {h[0], h[1]};
            size_ = d.x * d.y * d.z;
            dev_ptr_ = gpuMem::make_cuniquePtr<T>(size_);
        }

        /* create device array from partition */
        DeviceArray(const Partition<T> &rhs)
            : dims_(rhs.dims()), halo_(make_int2(rhs.halo()[0], rhs.halo()[1])), size_(rhs.size()),
              dev_ptr_(gpuMem::make_cuniquePtr<T>(rhs.size())) {
            SAFE_CALL(cudaMemcpy(dev_ptr_.get(), rhs.begin(), rhs.bytes(), cudaMemcpyHostToDevice));
        }

        // cuniquePtr makes destructor redundant
        ~DeviceArray() = default;

        //  copy constructor
        DeviceArray(const DeviceArray<T> &rhs)
            : dims_(rhs.dims_), halo_(rhs.halo_), size_(rhs.size_),
              dev_ptr_(gpuMem::make_cuniquePtr<T>(size_)) {
            SAFE_CALL(cudaMemcpy(dev_ptr_.get(), rhs.dev_ptr_.get(), rhs.bytes(),
                                 cudaMemcpyDeviceToDevice));
        }

        // assignment operator
        DeviceArray<T> &operator=(const DeviceArray &rhs) {
            if (this != &rhs) {
                dims_ = rhs.dims_;
                size_ = rhs.size_;
                halo_ = rhs.halo_;
                std::cout << "size_ = " << rhs.size_ << std::endl;
                dev_ptr_ = gpuMem::make_cuniquePtr<T>(rhs.size_);
                SAFE_CALL(cudaMemcpy(dev_ptr_.get(), rhs.dev_ptr_.get(), rhs.bytes(),
                                     cudaMemcpyDeviceToDevice));
            }
            return *this;
        }

        // move constructor
        DeviceArray(DeviceArray<T> &&rhs) = default;

        // move assignment operator
        DeviceArray<T> &operator=(DeviceArray<T> &&rhs) = default;

#ifdef __NVCC__
        // encapsulate raw ptr in DeviceMemory for device access
        operator gpu::DeviceMemory<T>() {
            return gpu::DeviceMemory<T>(dims_, halo_, dev_ptr_.get());
        }

        operator gpu::DeviceMemory<T>() const {
            return gpu::DeviceMemory<T>(dims_, halo_, dev_ptr_.get());
        }
#endif

        // access to the device-pointer
        T *dev_ptr() { return dev_ptr_.get(); }
        T *data() { return dev_ptr_.get(); }

        // access to the device-pointer
        const T *dev_ptr() const { return dev_ptr_.get(); };
        const T *data() const { return dev_ptr_.get(); }

        // size of the array
        [[nodiscard]] size_t size() const { return size_; }

        // bytes of the array
        [[nodiscard]] size_t bytes() const { return sizeof(T) * size_; }

        // get array dims
        [[nodiscard]] dim3_t dims() const { return dims_; }

        // get number of slices
        [[nodiscard]] int nslices() const { return dims_.x; }

        // get number of rows
        [[nodiscard]] int nrows() const { return dims_.y; }

        // get number of columns
        [[nodiscard]] int ncols() const { return dims_.z; }

        // initialize
        void init(T v) { gpu::init_array<T>(dev_ptr_.get(), v, size_); }

        // copy to partition
        void copy_to(Partition<T> &rhs) const {
            if (dims_ == rhs.dims()) {
                SAFE_CALL(
                    cudaMemcpy(rhs.begin(), dev_ptr_.get(), rhs.bytes(), cudaMemcpyDeviceToHost));
            } else {
                throw std::runtime_error("Partition and DeviceArray dimensions do not match");
            }
        }

        // copy to host
        std::vector<T> copy_to_host() const {
            std::vector<T> h_ptr(size_);
            SAFE_CALL(cudaMemcpy(h_ptr.data(), dev_ptr_.get(), bytes(), cudaMemcpyDeviceToHost));
            return h_ptr;
        }

        // operator overloading
        DeviceArray<T> operator*(const DeviceArray<T> &rhs) const {
            DeviceArray<T> res(dims_);
            gpu::multiply_arrays<T>(dev_ptr_.get(), rhs.dev_ptr_.get(), res.dev_ptr_.get(), size_);
            return res;
        }

        // multiply (for FFT Convolution)
        DeviceArray<T> multiply(const DeviceArray<T> &arr) const {

            DeviceArray<T> res(dims_);
            if ((arr.dims_.x != 1) || (arr.dims_.y != dims_.y) || (arr.dims_.z != dims_.z))
                throw std::runtime_error("Array is not a point spread function");
            gpu::broadcast_multiply<T>(dev_ptr_.get(), arr.dev_ptr_.get(), res.dev_ptr_.get(),
                                       dims_);
            return res;
        }

        // division (for normalization)
        DeviceArray<T> operator/(const T val) const {
            DeviceArray<T> res(dims_);
            if (val == static_cast<T>(0)) {
                throw std::runtime_error("division by zero");
            }
            T val_inv = static_cast<T>(1) / val;
            gpu::scale_array<T>(dev_ptr_.get(), val_inv, res.dev_ptr_.get(), size_);
            return res;
        }

        // /= operator
        DeviceArray<T> &operator/=(const T val) {
            if (val == static_cast<T>(0)) {
                throw std::runtime_error("division by zero");
            }
            T val_inv = static_cast<T>(1) / val;
            gpu::scale_array<T>(dev_ptr_.get(), val_inv, dev_ptr_.get(), size_);
            return *this;
        }

        // addition
        DeviceArray<T> operator+(const DeviceArray<T> &arr) const {
            DeviceArray<T> res(dims_);
            if (dims_ == arr.dims_)
                gpu::add_arrays<T>(dev_ptr_.get(), arr.dev_ptr_.get(), res.dev_ptr_.get(), size_);
            else
                throw std::runtime_error("Array dimensions do not match in +");
            return res;
        }

        // subtraction
        DeviceArray<T> operator-(const DeviceArray<T> &arr) const {
            DeviceArray<T> res(dims_);
            if (dims_ == arr.dims_)
                gpu::subtract_arrays<T>(dev_ptr_.get(), arr.dev_ptr_.get(), res.dev_ptr_.get(),
                                        size_);
            else {
                throw std::runtime_error("Array dimensions do not match in -");
            }
            return res;
        }

        // dot product
        T dot(const DeviceArray<T> &arr) const {
            if (dims_ != arr.dims_)
                throw std::runtime_error("Array dimensions do not match in dot");
            return gpu::dot(dev_ptr_.get(), arr.dev_ptr_.get(), size_);
        }

        // norm2
        T norm2() const { return gpu::dot<T>(dev_ptr_.get(), dev_ptr_.get(), size_); }
    };

    typedef DeviceArray<float> DeviceArrayf;
    typedef DeviceArray<double> DeviceArrayd;
    typedef DeviceArray<gpu::complex_t<float>> DeviceArraycf;
    typedef DeviceArray<gpu::complex_t<double>> DeviceArraycd;

    // conversion from complex to real
    template <typename T> DeviceArray<T> real(const DeviceArray<gpu::complex_t<T>> &arr) {
        DeviceArray<T> res(arr.dims());
        gpu::cast_array_to_real<T>(arr.dev_ptr(), res.dev_ptr(), arr.size());
        return res;
    }

    // conversion from real to complex
    template <typename T> DeviceArray<gpu::complex_t<T>> complex(const DeviceArray<T> &arr) {
        DeviceArray<gpu::complex_t<T>> res(arr.dims());
        gpu::cast_array_to_complex<T>(arr.dev_ptr(), res.dev_ptr(), arr.size());
        return res;
    }

} // namespace tomocam

#endif // TOMOCAM_DEV_ARRAY__H
