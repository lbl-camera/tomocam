/* -------------------------------------------------------------------------------
* Tomocam Copyright (c) 2018
*
* The Regents of the University of California, through Lawrence Berkeley National
* Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
*  Energy). All rights reserved.
*
* If you have questions about your rights to use or distribute this software,
* please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
*
* NOTICE. This Software was developed under funding from the U.S. Department of
* Energy and the U.S. Government consequently retains certain rights. As such, the
* U.S. Government has been granted for itself and others acting on its behalf a
* paid-up, nonexclusive, irrevocable, worldwide license in the Software to
* reproduce, distribute copies to the public, prepare derivative works, and
* perform publicly and display publicly, and to permit other to do so.
*---------------------------------------------------------------------------------
*/

#ifndef TOMOCAM_DISTARRAY__H
#define TOMOCAM_DISTARRAY__H

#include <vector>

namespace tomocam {

    struct dim3_t {
        unsigned x, y, z;
        dim3_t(const unsigned *d) : x(d[0]), y(d[1]), z(d[2]) {}
        dim3_t(const unsigned d0, const unsigned d1, const unsigned d2)
            : x(d0), y(d1), z(d2) {}
    };

    template <typename T>
    class Partition {
      private:
        dim3_t   dims_;
        T *      first_;
        unsigned size_;

      public:
        Partition()
            : dims_({0, 0, 0}), first_(nullptr), size_(0),
              device_id_(-1) {}
        Partition(dim3_t d, T *pos) : dims_(d), first_(pos)  {
            size_      = dims_.x * dims_.y * dims_.z;
            device_id_ = -1;
        }

        dim3_t dims() const { return dims_; }
        unsigned size() const { return size_; }
        T *      begin() { return first_; }
        
    };

    template <typename T>
    class DArray {
      private:
        dim3_t dims_;    ///< [Slices, Rows, Colums]
        size_t size_;    ///< Size of the alloated array
        T *    buffer_;  ///< Pointer to data buffer
        bool   unified_; ///< Flag for unified memory support

        // return global index
        int idx_(int i, int j) { return (i * dims_.y + j); }
        int idx_(int i, int j, int k) {
            return (i * dims_.y * dims_.z + j * dims_.z + k);
        }

      public:
        DArray(unsigned *);
        DArray(unsigned, unsigned, unsigned);
        DArray(dim3_t);
        ~DArray();

        // Forbid copy and move
        DArray(const DArray &) = delete;
        DArray(DArray &&)      = delete;
        DArray operator=(const DArray &) = delete;
        DArray operator=(DArray &&) = delete;

        // setup partitioning of array along slowest axis
        void create_partitions(unsigned);

        // getters
        /// dimensions of the array
        dim3_t dims() const { return dims_; };
        size_t slices() const { return dims_.x; }
        size_t rows() const { return dims_.y; }
        size_t cols() const {return dims_.z}

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
