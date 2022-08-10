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

#include "types.h"
#include "common.h"
#include "partition.h"

namespace tomocam {

    template <typename T>
    class DArray {
      private:
        dim3_t dims_; ///< [Slices, Rows, Colums]
        uint64_t size_;    ///< Size of the alloated array
        T *buffer_;   ///< Pointer to data buffer

        // return global index
        uint64_t idx_(int i, int j, int k) { 
            return (i * dims_.y * static_cast<uint64_t>(dims_.z) + 
            j * static_cast<uint64_t>(dims_.z) + k); 
        }

      public:
        DArray(dim3_t);
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

        // paste data to a buffer
        void paste(T *buf) const {
            #pragma omp parallel for
            for (uint64_t i = 0; i < size_; i++)
                buf[i] = buffer_[i];
        }

        // copy data from a buffer
        void copy(T *values) {
            #pragma omp parallel for
            for (uint64_t i = 0; i < size_; i++)
                buffer_[i] = values[i];
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
            return std::sqrt(v);
        }

        // sum
        T sum() const {
            T v = 0;
            #pragma omp parallel for reduction( + : v)
            for (uint64_t i = 0; i < size_; i++) 
                v += buffer_[i];
            return v;
        }

        T max() const {
            T v = -1E10;
            #pragma omp parallel for reduction(max : v)
            for (uint64_t i = 0; i < size_; i++)
                if (buffer_[i] > v) 
                    v = buffer_[i];
            return v;
        }

        T min() const {
            T v = 1E10;
            #pragma omp parallel for reduction(min : v)
            for (uint64_t i = 0; i < size_; i++)
                if (buffer_[i] < v) 
                    v = buffer_[i];
            return v;
        }


        // arithmatic operators
        DArray<T> operator+(const DArray<T> &);
        DArray<T> operator-(const DArray<T> &);
        DArray<T> &operator+=(const DArray<T> &);
        DArray<T> operator*(const DArray<T> &);
        DArray<T> operator*(T );

        // save array to file
        void to_file(const char * filename) {
            std::ofstream fout(filename, std::ios::binary);
            fout.write((char *) buffer_, this->bytes());
            fout.close();
        }

        /// dimensions of the array
        dim3_t dims() const { return dims_; };
        uint64_t slices() const { return dims_.x; }
        uint64_t rows() const { return dims_.y; }
        uint64_t cols() const { return dims_.z; }
        uint64_t size() const { return size_; }
        size_t bytes() const { return size_ * sizeof(T); }

        // indexing
        T &operator[](uint64_t i) { return buffer_[i]; }
        T &operator()(int i, int j, int k) { return buffer_[idx_(i, j, k)]; }

        // padded
        T padded(int i, int j, int k) {
            if ((i < 0) || (i >= dims_.x) || 
                    (j < 0) || (j >= dims_.y) ||
                    (k < 0) || (k >= dims_.z)) 
                return 0;
            else
                return buffer_[idx_(i, j, k)]; 
        }

        /// Returns pointer to N-th slice
        T *slice(int n) { return (buffer_ + n * dims_.y * dims_.z); }

        /// Expose the alloaated memoy pointer
        T *data() { return buffer_; }

    };
} // namespace tomocam

#include "dist_array.cpp"
#endif // TOMOCAM_DISTARRAY__H
