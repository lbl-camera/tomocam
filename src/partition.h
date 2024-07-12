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

#ifndef TOMOCAM_PARTITION__H
#define TOMOCAM_PARTITION__H

#include <vector>

#include "types.h"
#include "common.h"

namespace tomocam {

    template <typename T>
    class Partition {
      private:
        dim3_t dims_;
        uint64_t size_;
        T *first_;
        int halo_[2];

      public:
        Partition(dim3_t d, T *pos) : dims_(d), first_(pos) {
            size_ = static_cast<uint64_t>(dims_.z) * dims_.y * dims_.x; 
            halo_[0] = 0;
            halo_[1] = 0;
        }

        Partition(dim3_t d, T *pos, int *h) : dims_(d), first_(pos) {
            size_ = static_cast<uint64_t>(dims_.z) * dims_.y * dims_.x; 
            halo_[0] = h[0];
            halo_[1] = h[1];
        }

        dim3_t dims() const { return dims_; }
        int nslices() const { return dims_.x; }
        int nrows() const { return dims_.y; }
        int ncols() const { return dims_.z; } 
        uint64_t size() const { return size_; }
        size_t bytes() const { return size_ * sizeof(T); }
        int  *halo() { return halo_; }
        const int  *halo() const { return halo_; }

        T *begin() { return first_; }
        const T *begin() const { return first_; }
        T *slice(int i) { return first_ + i * dims_.y * dims_.z; }
        const T *slice(int i) const { return first_ + i * dims_.y * dims_.z; }

        T &operator()(int i, int j, int k) {
            return first_[i * dims_.y * dims_.z + j * dims_.z + k];
        }
        T operator()(int i, int j, int k) const {
            return first_[i * dims_.y * dims_.z + j * dims_.z + k];
        }
    };

    // partition an array into sub-partitions
    template <typename T, template <typename> class Array>
    std::vector<Partition<T>> create_partitions(Array<T> &a, int npartitions) {

        // create sub-partions
        std::vector<Partition<T>> table;
        auto dims = a.dims();

        int xparts = dims.x / npartitions;
        int nextra = dims.x % npartitions;
        int offset = 0;
        for (int i = 0; i < npartitions; i++) {
            dim3_t d(xparts, dims.y, dims.z);
            if (i < nextra) d.x += 1;
            table.push_back(Partition<T>(d, a.slice(offset)));
            offset += d.x;
        }
        return table;
    }

    // partition an array into sub-partitions with halo in x-direction
    template <typename T, template <typename> class Array>
    std::vector<Partition<T>> create_partitions(Array<T> &a, int npartitions,
        int halo) {

        // partition the array into sub-partitions with halo in x-direction
        std::vector<Partition<T>> table;
        auto dims = a.dims();

        // check if the array has halo member-function
        constexpr bool has_halo = requires(const Array<T> &a) { a.halo(); };

        // get existing halo, if any
        int offset;
        int ahalo[2];
        if constexpr (has_halo) {
            ahalo[0] = a.halo()[0];
            ahalo[1] = a.halo()[1];
            offset = std::max(ahalo[0] - halo, 0);
        } else {
            ahalo[0] = 0;
            ahalo[1] = 0;
            offset = 0;
        }

        // actual number of slices
        int nslcs = dims.x - ahalo[0] - ahalo[1];
        int xparts = nslcs / npartitions;
        int nextra = nslcs % npartitions;

        // make first partition.
        int nx = xparts + halo;
        int h[2] = {0, halo};
        if (ahalo[0] > 0) {
            nx += halo;
            h[0] = halo;
        }
        if (nextra > 0) nx += 1;
        dim3_t d(nx, dims.y, dims.z);
        table.push_back(Partition<T>(d, a.slice(offset), h));
        offset = xparts;
        if (nextra > 0) offset += 1;
        if (ahalo[0] > 0) offset += halo;

        // create the rest of the partitions, but the last one
        for (int i = 1; i < npartitions - 1; i++) {
            dim3_t d(xparts + 2 * halo, dims.y, dims.z);
            if (i < nextra) d.x += 1;
            h[0] = halo;
            h[1] = halo;
            table.push_back(Partition<T>(d, a.slice(offset - halo), h));
            offset += xparts;
            if (i < nextra) offset += 1;
        }

        // make the last partition. If there is halo at the end, keep it
        nx = xparts + halo;
        h[0] = halo;
        h[1] = 0;
        if (ahalo[1] > 0) {
            nx += halo;
            h[1] = halo;
        }
        dim3_t d2(nx, dims.y, dims.z);
        table.push_back(Partition<T>(d2, a.slice(offset - halo), h));
        return table;
    }
} // namespace tomocam

#endif // TOMOCAM_PARTITION__H

