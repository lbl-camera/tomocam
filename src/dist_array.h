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


#include "types.h"
#include "common.h"

namespace tomocam {

    template <typename T>
    class Partition {
      private:
        dim3_t dims_;
        int size_;
        T *first_;
        int halo_[2];

      public:
        Partition(dim3_t d, T *pos) : dims_(d), first_(pos) {
            size_ = dims_.x * dims_.y * dims_.z; 
            halo_[0] = 0;
            halo_[1] = 0;
        }

        Partition(dim3_t d, T *pos, int *h) : dims_(d), first_(pos) {
            size_ = dims_.x * dims_.y * dims_.z; 
            halo_[0] = h[0];
            halo_[1] = h[1];
        }

        dim3_t dims() const { return dims_; }
        int size() const { return size_; }
        size_t bytes() const { return size_ * sizeof(T); }
        int  *halo() { return halo_; }

        T *begin() { return first_; }
        T *slice(int i) { return first_ + i * dims_.y * dims_.z; }

        // create sub-partions 
        std::vector<Partition<T>> sub_partitions(int);
        std::vector<Partition<T>> sub_partitions(int, int);
    };

    template <typename T>
    class DArray {
      private:
        bool owns_buffer_; ///< Don't free buffer if not-owned
        dim3_t dims_; ///< [Slices, Rows, Colums]
        int size_;    ///< Size of the alloated array
        T *buffer_;   ///< Pointer to data buffer

        // return global index
        int idx_(int i, int j, int k) { return (i * dims_.y * dims_.z + j * dims_.z + k); }

      public:
        DArray(dim3_t);
        DArray(np_array_t<T>);
        ~DArray();

        //  copy and move
        DArray(const DArray &);
        DArray& operator=(const DArray &);
        DArray(DArray &&);
        DArray& operator=(DArray &&);

        // setup partitioning of array along slowest dimension
        std::vector<Partition<T>> create_partitions(int);

        // create partitionng along slowest dimension with halo
        std::vector<Partition<T>> create_partitions(int, int);

        // copy data to DArray
        void init(T *values) {
            #pragma omp parallel for
            for (int i = 0; i < size_; i++)
                buffer_[i] = values[i];
        }

        // init
        void init(T v) {
            #pragma omp parallel for
            for (int i = 0; i < size_; i++)
                buffer_[i] = v;
        }

        // norm2
        T norm() const {
            T v = 0;
            #pragma omp parallel for reduction( + : v)
                for (int i = 0; i < size_; i++)
                    v += buffer_[i] * buffer_[i];
            return std::sqrt(v);
        }

        // sum
        T sum() const {
            T v = 0;
            #pragma omp parallel for reduction( + : v)
            for (int i = 0; i < size_; i++) v += buffer_[i];
            return v;
        }

        T max() const {
            T v = -1E10;
            #pragma omp parallel for reduction(max : v)
            for (int i = 0; i < size_; i++)
                if (buffer_[i] > v) v = buffer_[i];
            return v;
        }

        T min() const {
            T v = 1E10;
            #pragma omp parallel for reduction(min : v)
            for (int i = 0; i < size_; i++)
                if (buffer_[i] < v) v = buffer_[i];
            return v;
        }

        /// dimensions of the array
        dim3_t dims() const { return dims_; };
        int slices() const { return dims_.x; }
        int rows() const { return dims_.y; }
        int cols() const { return dims_.z; }
        int size() const { return size_; }
        size_t bytes() const { return size_ * sizeof(T); }

        // indexing
        T &operator[](int i) { return buffer_[i]; }
        T &operator()(int i, int j, int k) { return buffer_[idx_(i, j, k)]; }

        /// Returns pointer to N-th slice
        T *slice(int n) { return (buffer_ + n * dims_.y * dims_.z); }

        /// Expose the alloaated memoy pointer
        T *data() { return buffer_; }

    };
} // namespace tomocam

#include "dist_array.cpp"
#endif // TOMOCAM_DISTARRAY__H
