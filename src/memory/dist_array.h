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

#ifndef TOMOCAM_DISTARRAY__H
#define TOMOCAM_DISTARRAY__H

#include <algorithm>
#include <execution>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "memory/common.h"
#include "memory/partition.h"
#include "utils/types.h"

#ifdef MULTIPROC
#include "concurrency/multiproc.h"
#endif

namespace tomocam {

    namespace exe = std::execution;

    template <typename T>
    class DArray {
      private:
        dim3_t dims_;
        size_t size_;
        std::unique_ptr<T[]> buffer_;

        // return global index
        size_t idx_(size_t i, size_t j, size_t k) const {
            return (i * dims_.y + j) * dims_.z + k;
        }

      public:
        // only constructor
        DArray(dim3_t d) : dims_(d) {
            size_ = static_cast<size_t>(d.z * d.y * d.x);
            buffer_ = std::make_unique<T[]>(size_);
        }

        // destructor
        ~DArray() = default;

        // delete copy constructor and assignment operator
        DArray(const DArray &rhs) = delete;
        DArray &operator=(const DArray &rhs) = delete;

        // move constructor
        DArray(DArray &&rhs) = default;

        // move assignment operator
        DArray &operator=(DArray &&rhs) = default;

        // rust-like clone method
        DArray clone() const {
            DArray result(dims_);
            std::copy(this->begin(), this->end(), result.begin());
            return result;
        }

        void match_dims(const dim3_t &d) {
            if (dims_.x != d.x || dims_.y != d.y || dims_.z != d.z) {
                std::runtime_error("Error: DArray dimensions do not match");
            }
        }

        // init
        void init(T v) { std::fill(exe::par_unseq, this->begin(), this->end(), v); }

        // inplace subtract
        DArray<T> &operator-=(const DArray<T> &rhs) {
            match_dims(rhs.dims_);
            std::transform(exe::par_unseq, this->begin(), this->end(), rhs.begin(),
                           this->begin(), std::minus<T>());
            return *this;
        }

        // inplace add
        DArray<T> &operator+=(const DArray<T> &rhs) {
            match_dims(rhs.dims_);
            std::transform(exe::par_unseq, this->begin(), this->end(), rhs.begin(),
                           this->begin(), std::plus<T>());
            return *this;
        }

        // inplace multiply
        DArray<T> &operator*=(const DArray<T> &rhs) {
            match_dims(dims_);
            std::transform(exe::par_unseq, this->begin(), this->end(), rhs.begin(),
                           this->begin(), std::multiplies<T>());
            return *this;
        }

        // inplace divide
        DArray<T> &operator/=(const DArray<T> &rhs) {
            match_dims(rhs.dims_);
            std::transform(exe::par_unseq, this->begin(), this->end(), rhs.begin(),
                           this->begin(), std::divides<T>());
            return *this;
        }

        // inplace subtract a scalar
        DArray<T> &operator-=(const T &rhs) {
            std::transform(exe::par_unseq, this->begin(), this->end(), this->begin(),
                           [rhs](T a) { return a - rhs; });
            return *this;
        }

        // inplace add a scalar
        DArray<T> &operator+=(const T &rhs) {
            std::transform(exe::par_unseq, this->begin(), this->end(), this->begin(),
                           [rhs](T a) { return a + rhs; });
            return *this;
        }

        // inplace multiply by a scalar
        DArray<T> &operator*=(const T &rhs) {
            std::transform(exe::par_unseq, this->begin(), this->end(), this->begin(),
                           [rhs](T a) { return a * rhs; });
            return *this;
        }

        // inplace divide by a scalar
        DArray<T> &operator/=(const T &rhs) {
            if (rhs == T(0)) {
                throw std::runtime_error(
                    "Error: Division by zero in DArray::operator/=");
            }
            std::transform(exe::par_unseq, this->begin(), this->end(), this->begin(),
                           [rhs](T a) { return a / rhs; });
            return *this;
        }

        /// dimensions of the array
        dim3_t dims() const { return dims_; };
        int nslices() const { return dims_.x; }
        int nrows() const { return dims_.y; }
        int ncols() const { return dims_.z; }
        size_t size() const { return size_; }
        size_t bytes() const { return size_ * sizeof(T); }

        // indexing
        T &operator[](size_t i) { return buffer_.get()[i]; }
        T operator[](size_t i) const { return buffer_.get()[i]; }

        T &operator()(int i, int j, int k) {
            size_t ii = static_cast<size_t>(i);
            size_t jj = static_cast<size_t>(j);
            size_t kk = static_cast<size_t>(k);
            return buffer_.get()[idx_(ii, jj, kk)];
        }
        const T &operator()(int i, int j, int k) const {
            size_t ii = static_cast<size_t>(i);
            size_t jj = static_cast<size_t>(j);
            size_t kk = static_cast<size_t>(k);
            return buffer_.get()[idx_(ii, jj, kk)];
        }

        // Returns pointer to N-th slice
        T *slice(int n) { return (buffer_.get() + n * dims_.y * dims_.z); }
        const T *slice(int n) const {
            return (buffer_.get() + n * dims_.y * dims_.z);
        }

        // Expose the allocated memoy pointer
        T *begin() { return buffer_.get(); }
        const T *begin() const { return buffer_.get(); }

        // Expose the end of the allocated memory
        T *end() { return buffer_.get() + size_; }
        const T *end() const { return buffer_.get() + size_; }

#ifdef MULTIPROC
        void update_neigh_proc() {
            // set up neighs
            int myrank = multiproc::mp.myrank();
            int nproc = multiproc::mp.nprocs();
            int prev = myrank - 1;
            int next = myrank + 1;
            bool first = multiproc::mp.first();
            bool last = multiproc::mp.last();

            // buffer size
            size_t count = dims_.y * dims_.z;

            // send slice 1 to prev
            if (!first) multiproc::mp.Send(this->slice(1), count, prev);

            // receive last slice from next
            if (!last) multiproc::mp.Recv(this->slice(dims_.x - 1), count, next);

            // send penultimate slice to next
            if (!last) multiproc::mp.Send(this->slice(dims_.x - 2), count, next);
            //
            // receive slice 0 from prev
            if (!first) multiproc::mp.Recv(this->slice(0), count, prev);
        }
#endif // MULTIPROC
    };

    /* subdivide array into N partitions */
    template <typename T>
    std::vector<Partition<T>> create_partitions(DArray<T> &arr, int n_partitions) {

        dim3_t dims = arr.dims();
        int n_slices = arr.nslices() / n_partitions;
        int n_extra = arr.nslices() % n_partitions;

        // vector to hold the partitions
        std::vector<Partition<T>> table;
        int offset = 0;
        for (int i = 0; i < n_partitions; i++) {
            if (i < n_extra)
                dims.x = n_slices + 1;
            else
                dims.x = n_slices;
            table.push_back(Partition<T>(dims, arr.slice(offset)));
            offset += dims.x;
        }
        return table;
    }

/* subdivide array into N partitions, with n halo layers on boundaries */
#ifndef MULTIPROC
    template <typename T>
    std::vector<Partition<T>> create_partitions(DArray<T> &arr, int n_partitions,
                                                int halo) {

        const dim3_t dims = arr.dims();
        int n_slices = arr.nslices() / n_partitions;
        int n_extra = arr.nslices() % n_partitions;

        // vector to hold the partitions
        std::vector<Partition<T>> table;
        std::vector<int> locations;

        // create partition locations as if there were no halo layers
        int offset = 0;
        locations.push_back(offset);
        for (int i = 0; i < n_partitions; i++) {
            if (i < n_extra)
                offset += n_slices + 1;
            else
                offset += n_slices;
            locations.push_back(offset);
        }

        // add halo layers
        int h[2];
        for (int i = 0; i < n_partitions; i++) {
            int imin = std::max(locations[i] - halo, 0);
            if (i == 0)
                h[0] = 0;
            else
                h[0] = halo;
            int imax = std::min(locations[i + 1] + halo, dims.x);
            if (i == n_partitions - 1)
                h[1] = 0;
            else
                h[1] = halo;
            dim3_t d(imax - imin, dims.y, dims.z);
            table.push_back(Partition<T>(d, arr.slice(imin), h));
        }
        return table;
    }
#else
    template <typename T>
    std::vector<Partition<T>> create_partitions(DArray<T> &arr, int n_partitions,
                                                int halo) {

        // get MPI rank and size
        int myrank = multiproc::mp.myrank();
        int nprocs = multiproc::mp.nprocs();

        const dim3_t dims = arr.dims();
        int work = dims.x;
        if (myrank > 0) work -= 1;
        if (myrank < nprocs - 1) work -= 1;

        // subdivide actual work
        int n_slices = work / n_partitions;
        int n_extra = work % n_partitions;

        // vector to hold the partitions
        std::vector<Partition<T>> table;
        std::vector<int> locations;

        // create partition locations
        int offset = myrank == 0 ? 0 : 1;
        locations.push_back(offset);
        for (int i = 0; i < n_partitions; i++) {
            if (i < n_extra)
                offset += n_slices + 1;
            else
                offset += n_slices;
            locations.push_back(offset);
        }

        // add halo layers
        int h[2];
        for (int i = 0; i < n_partitions; i++) {
            int imin = std::max(locations[i] - halo, 0);
            if ((myrank == 0) && (i == 0))
                h[0] = 0;
            else
                h[0] = halo;
            int imax = std::min(locations[i + 1] + halo, dims.x);
            if ((myrank == nprocs - 1) && (i == n_partitions - 1))
                h[1] = 0;
            else
                h[1] = halo;
            dim3_t d(imax - imin, dims.y, dims.z);
            table.push_back(Partition<T>(d, arr.slice(imin), h));
        }
        return table;
    }
#endif
} // namespace tomocam
#endif // TOMOCAM_DISTARRAY__H
