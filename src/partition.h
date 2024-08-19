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

            dim3_t dims() const {
                return dims_;
            }
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
            int  *halo() {
                return halo_;
            }
            const int  *halo() const {
                return halo_;
            }

            T *begin() {
                return first_;
            }
            const T *begin() const {
                return first_;
            }
            T *slice(int i) {
                return first_ + i * dims_.y * dims_.z;
            }
            const T *slice(int i) const {
                return first_ + i * dims_.y * dims_.z;
            }

            T &operator()(int i, int j, int k) {
                return first_[i * dims_.y * dims_.z + j * dims_.z + k];
            }
            T operator()(int i, int j, int k) const {
                return first_[i * dims_.y * dims_.z + j * dims_.z + k];
            }
    };

    // partition an array into sub-partitions
    template <typename T>
    std::vector<Partition<T>> create_partitions(Partition<T> &a,
        int npartitions) {

        // create sub-partitions
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
    template <typename T>
    std::vector<Partition<T>> create_partitions(Partition<T> &a,
        int npartitions, int halo) {

        // sanity check
        if (npartitions < 0) {
            throw std::runtime_error("Number of partitions must be at least 1");
        }

        // sanity check
        if (npartitions < 0) {
            throw std::runtime_error("Number of partitions must be at least 1");
        }

        // partition the array into sub-partitions with halo in x-direction
        std::vector<Partition<T>> table;
        auto dims = a.dims();
        int h[2] = {0, 0};

        // if there is only one partition, return the whole array
        if (npartitions == 1) {
            table.push_back(Partition<T>(dims, a.begin(), a.halo()));
            return table;
        }

        // actual number of slices
        int offset = a.halo()[0];
        int nslcs = dims.x - a.halo()[0] - a.halo()[1];
        int work = nslcs / npartitions;
        int extra = nslcs % npartitions;

        if (work == 0) {
            throw std::runtime_error(
                "Number of partitions is too large for the array");
        }

        // set pointers as if  there is no halo
        std::vector<int> shares(npartitions, work);
        for (int i = 0; i < extra; i++) {
            shares[i] += 1;
        }

        // create the sub-partitions
        int h[2];
        for (int i = 0; i < npartitions; i++) {
            if (i == 0) h[0] = a.halo()[0];
            else h[0] = halo;
            if (i == npartitions - 1) h[1] = a.halo()[1];
            else h[1] = halo;
            dim3_t d(shares[i] + h[0] + h[1], dims.y, dims.z);
            table.push_back(Partition<T>(d, a.slice(offset - h[0]), h));
            offset += shares[i];
        }
        return table;
    }
} // namespace tomocam

#endif // TOMOCAM_PARTITION__H
