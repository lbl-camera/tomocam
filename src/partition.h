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
        uint64_t size() const { return size_; }
        size_t bytes() const { return size_ * sizeof(T); }
        int  *halo() { return halo_; }

        T *begin() { return first_; }
        T *slice(int i) { return first_ + i * dims_.y * dims_.z; }

        // create sub-partions 
        std::vector<Partition<T>> sub_partitions(int nmax) {
            std::vector<Partition<T>> table;

            int offset = 0;
            int n_partitions = dims_.x / nmax;
            dim3_t d(nmax, dims_.y, dims_.z);
            for (int i = 0; i < n_partitions; i++) {
                table.push_back(Partition<T>(d, slice(offset)));
                offset += nmax;
            }
            int n_extra = dims_.x % nmax;
            if (n_extra > 0) {
                d.x = n_extra;
                table.push_back(Partition<T>(d, slice(offset)));
            }
            return table;
        }
    
        // and with halo, as well
        std::vector<Partition<T>> sub_partitions(int partition_size, int halo) {
            std::vector<Partition<T>> table;    

            int sub_halo[2];
            int slcs = dims_.x - halo_[0] - halo_[1];
            int n_partitions = slcs / partition_size;
            int n_extra = slcs % partition_size;
            std::vector<int> locations;

            for (int i = 0; i < n_partitions; i++)
                locations.push_back(halo_[0] + i * partition_size);
            locations.push_back(halo_[0] + n_partitions * partition_size);

            // if they don't divide nicely
            if (n_extra > 0) {
                locations.push_back(dims_.x);
                n_partitions += 1;
            }

            for (int i = 0; i < n_partitions; i++) {
                int imin = std::max(locations[i] - halo, 0);
                // check if it is an end
                if ((i == 0) && (halo_[0] == 0)) 
                    sub_halo[0] = 0;
                else 
                    sub_halo[0] = halo;

                int imax = std::min(locations[i+1] + halo, dims_.x);
                // check if it is an end
                if ((i == n_partitions-1) && (halo_[1] == 0))
                    sub_halo[1] = 0;
                else 
                    sub_halo[1] = halo;

                dim3_t d(imax-imin, dims_.y, dims_.z);
                table.push_back(Partition<T>(d, slice(imin), sub_halo));
            }
            return table;
        }
    };
} // namespace tomocam

#endif // TOMOCAM_PARTITION__H

