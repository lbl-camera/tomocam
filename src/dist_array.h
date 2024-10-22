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

#include <vector>
#include <fstream>
#include <iostream>
#include <format>

#include "types.h"
#include "common.h"
#include "partition.h"

#ifdef MULTIPROC
#include "multiproc.h"
#endif

namespace tomocam {

    template <typename T>
    class DArray {
        private:
            dim3_t dims_;      ///< [Slices, Rows, Colums]
            uint64_t size_;    ///< Size of the alloated array
            T *buffer_;        ///< Pointer to data buffer

            // return global index
            uint64_t idx_(int i, int j, int k) const {
                uint64_t z64 = static_cast<uint64_t>(dims_.z);
                return (i * dims_.y * z64 + j * z64 + k);
            }

        public:
            // only constructor
            DArray(dim3_t d) : dims_(d) {
                size_ = static_cast<uint64_t>(d.z) * d.y * d.x;
                buffer_ = new T[size_];
            }

            // destructor
            ~DArray() {
                delete[] buffer_;
            }

            //  copy constructor
            DArray(const DArray &rhs) {
                dims_ = rhs.dims_;
                size_ = rhs.size_;
                buffer_ = new T[size_];
                std::copy(rhs.buffer_, rhs.buffer_ + size_, buffer_);
            }

            // assignment operator
            DArray &operator=(const DArray &rhs) {
                if (this != &rhs) {
                    dims_ = rhs.dims_;
                    size_ = rhs.size_;
                    delete[] buffer_;
                    buffer_ = new T[size_];
                    std::copy(rhs.buffer_, rhs.buffer_ + size_, buffer_);
                }
                return *this;
            }

            // move constructor
            DArray(DArray &&rhs) {
                dims_ = rhs.dims_;
                size_ = rhs.size_;
                buffer_ = rhs.buffer_;
                rhs.buffer_ = nullptr;
            }

            // move assignment operator
            DArray &operator=(DArray &&rhs) {
                if (this != &rhs) {
                    dims_ = rhs.dims_;
                    size_ = rhs.size_;
                    delete[] buffer_;
                    buffer_ = rhs.buffer_;
                    rhs.buffer_ = nullptr;
                }
                return *this;
            }

            // init
            void init(T v) {
                #pragma omp parallel for
                for (uint64_t i = 0; i < size_; i++)
                    buffer_[i] = v;
            }

            // norm2
            T norm() const {
                T v = 0;
                #pragma omp parallel for reduction( + : v)
                for (uint64_t i = 0; i < size_; i++)
                    v += buffer_[i] * buffer_[i];
                return v;
            }

            // sum
            T sum() const {
                T v = 0;
                #pragma omp parallel for reduction( + : v)
                for (uint64_t i = 0; i < size_; i++)
                    v += buffer_[i];
                return v;
            }

            // max
            T max() const {
                T v = 1.0e-20;
                #pragma omp parallel for reduction(max : v)
                for (uint64_t i = 0; i < size_; i++)
                    if (buffer_[i] > v)
                        v = buffer_[i];
                return v;
            }

            // min
            T min() const {
                T v = 1.0e20;
                #pragma omp parallel for reduction(min : v)
                for (uint64_t i = 0; i < size_; i++)
                    if (buffer_[i] < v)
                        v = buffer_[i];
                return v;
            }

            // subtract
            DArray<T> operator-(const DArray<T> &rhs) const {
                DArray<T> out(dims_);
                #pragma omp parallel for
                for (uint64_t i = 0; i < size_; i++)
                    out.buffer_[i] = buffer_[i] - rhs.buffer_[i];
                return out;
            }

            // add
            DArray<T> operator+(const DArray<T> &rhs) const {
                DArray<T> out(dims_);
                #pragma omp parallel for
                for (uint64_t i = 0; i < size_; i++)
                    out.buffer_[i] = buffer_[i] + rhs.buffer_[i];
                return out;
            }

            // multiply
            DArray<T> operator*(T v) const {
                DArray<T> out(dims_);
                #pragma omp parallel for
                for (uint64_t i = 0; i < size_; i++)
                    out.buffer_[i] = buffer_[i] * v;
                return out;
            }

            // drop a column
            void dropcol() {
                dim3_t d = {dims_.x, dims_.y, dims_.z - 1};
                uint64_t new_size = static_cast<uint64_t>(d.z) * d.y * d.x;
                T *new_buffer = new T[new_size];

                // copy data
                uint64_t nz = static_cast<uint64_t>(dims_.z);
                uint64_t new_nz = static_cast<uint64_t>(d.z);
                #pragma omp parallel for
                for (int i = 0; i < dims_.x; i++) {
                    for (int j = 0; j < dims_.y; j++) {
                        uint64_t beg = i * dims_.y * nz + j * nz;
                        uint64_t end = i * dims_.y * nz + j * nz + d.z;
                        uint64_t out_beg = i * d.y * new_nz + j * new_nz;
                        std::copy(buffer_ + beg, buffer_ + end,
                            new_buffer + out_beg);
                    }
                }
                delete[] buffer_;
                buffer_ = new_buffer;
                new_buffer = nullptr;
                dims_ = d;
                size_ = new_size;
            }

            /// dimensions of the array
            dim3_t dims() const {
                return dims_;
            };
            int nslices() const {
                return dims_.x;
            }
            int nrows() const {
                return dims_.y;
            }
            int ncols() const {
                return dims_.z;
            }
            uint64_t size() const {
                return size_;
            }
            size_t bytes() const {
                return size_ * sizeof(T);
            }

            // indexing
            T &operator[](uint64_t i) {
                return buffer_[i];
            }
            T operator[](uint64_t i) const {
                return buffer_[i];
            }
            T &operator()(int i, int j, int k) {
                return buffer_[idx_(i, j, k)];
            }
            T operator()(int i, int j, int k) const {
                return buffer_[idx_(i, j, k)];
            }

            // Returns pointer to N-th slice
            T *slice(int n) {
                return (buffer_ + n * dims_.y * dims_.z);
            }
            const T *slice(int n) const {
                return (buffer_ + n * dims_.y * dims_.z);
            }

            // Expose the allocated memoy pointer
            T *begin() {
                return buffer_;
            }
            T *data() {
                return buffer_;
            }
            const T *begin() const {
                return buffer_;
            }
            const T *data() const {
                return buffer_;
            }

            // Expose the end of the allocated memory
            T *end() {
                return buffer_ + size_;
            }
            const T *end() const {
                return buffer_ + size_;
            }

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

    template <typename T>
    DArray<T> operator*(T v, const DArray<T> &rhs) {
        return rhs * v;
    }

    /* subdivide array into N partitions */
    template <typename T>
    std::vector<Partition<T>> create_partitions(DArray<T> &arr,
        int n_partitions) {

        dim3_t dims = arr.dims();
        int n_slices = arr.nslices() / n_partitions;
        int n_extra = arr.nslices() % n_partitions;

        // vector to hold the partitions
        std::vector<Partition<T>> table;
        int offset = 0;
        for (int i = 0; i < n_partitions; i++) {
            if (i < n_extra) dims.x = n_slices + 1;
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
    std::vector<Partition<T>> create_partitions(DArray<T> &arr,
        int n_partitions, int halo) {

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
            if (i < n_extra) offset += n_slices + 1;
            else
                offset += n_slices;
            locations.push_back(offset);
        }

        // add halo layers
        int h[2];
        for (int i = 0; i < n_partitions; i++) {
            int imin = std::max(locations[i] - halo, 0);
            if (i == 0) h[0] = 0;
            else
                h[0] = halo;
            int imax = std::min(locations[i + 1] + halo, dims.x);
            if (i == n_partitions - 1) h[1] = 0;
            else
                h[1] = halo;
            dim3_t d(imax - imin, dims.y, dims.z);
            table.push_back(Partition<T>(d, arr.slice(imin), h));
        }
        return table;
    }
    #else
    template <typename T>
    std::vector<Partition<T>> create_partitions(DArray<T> &arr,
        int n_partitions, int halo) {

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
            if (i < n_extra) offset += n_slices + 1;
            else
                offset += n_slices;
            locations.push_back(offset);
        }

        // add halo layers
        int h[2];
        for (int i = 0; i < n_partitions; i++) {
            int imin = std::max(locations[i] - halo, 0);
            if ((myrank == 0) && (i == 0)) h[0] = 0;
            else
                h[0] = halo;
            int imax = std::min(locations[i + 1] + halo, dims.x);
            if ((myrank == nprocs - 1) && (i == n_partitions - 1)) h[1] = 0;
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
