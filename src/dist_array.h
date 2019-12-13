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

namespace tomocam {

    struct dim3_t {
        int x, y, z;
        dim3_t()
            : x(0), y(0), z(0) {}
        dim3_t(int d0, int d1, int d2)
            : x(d0), y(d1), z(d2) {}
    };

    template <typename T>
    class Partition {
      private:
        dim3_t dims_;
        T *    first_;

      public:
        Partition() : dims_({0, 0, 0}), first_(nullptr) {}
        Partition(dim3_t d, T *pos) : dims_(d), first_(pos) {} 
        dim3_t dims() const { return dims_; }
        T *    begin() { return first_; }
    };

    template <typename T>
    class DArray {
      private:
        dim3_t dims_;    ///< [Slices, Rows, Colums]
        int    size_;    ///< Size of the alloated array
        T *    buffer_;  ///< Pointer to data buffer

        // return global index
        int idx_(int i, int j) { return (i * dims_.y + j); }
        int idx_(int i, int j, int k) {
            return (i * dims_.y * dims_.z + j * dims_.z + k);
        }

      public:
        DArray() = delete;
        DArray(int, int, int);
        DArray(dim3_t);
        ~DArray();

        // Forbid copy and move
        DArray(const DArray &) = delete;
        DArray operator=(const DArray &) = delete;
        DArray(DArray &&)      = delete;
        DArray operator=(DArray &&) = delete;

        // setup partitioning of array along slowest axis
        std::vector<Partition<T>> create_partitions(int);

        // getters
        /// dimensions of the array
        dim3_t dims() const { return dims_; };
        int slices() const { return dims_.x; }
        int rows() const { return dims_.y; }
        int cols() const { return dims_.z; }
        int size() const { return size_; }

        // indexing
        T &operator()(int);
        T &operator()(int, int);
        T &operator()(int, int, int);

        /// Returns pointer to N-th slice
        T *slice(int n) { return (buffer_ + n * dims_.y * dims_.z); }

        /// Expose the alloaated memoy pointer
        T *data() { return buffer_; }
    };
} // namespace tomocam

#include "dist_array.cpp"
#endif // TOMOCAM_DISTARRAY__H
