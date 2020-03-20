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

#include <iostream>

namespace tomocam {


    // partition type can create sub_partitions
    template <typename T>
    std::vector<Partition<T>> Partition<T>::sub_partitions(int nmax) {
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
    template <typename T>
    std::vector<Partition<T>> Partition<T>::sub_partitions(int nmax, int halo) {
        std::vector<Partition<T>> table;    

        int sub_halo[2];
        int slcs = dims_.x - halo_[0] - halo_[1];
        int n_partitions = slcs / nmax;
        int n_extra = slcs % nmax;
        std::vector<int> locations;

        for (int i = 0; i < n_partitions; i++)
            locations.push_back(halo_[0] + i * nmax);
        locations.push_back(halo_[0] + n_partitions * nmax);

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
            

    /*
     *
     *
     * DistArray<T> definitions
     *
     *
     *
     */
    template <typename T>
    DArray<T>::DArray(dim3_t dim) {
        // limit ndims to 3
        int ndims = 3;
        dims_     = dim;
        size_     = dims_.x * dims_.y * dims_.z;

        // allocate memory for the array
        buffer_ = new T [size_];
    }
    // for calling from python
    template <typename T>
    DArray<T>::DArray(int nx, int ny, int nz) {
        // limit ndims to 3
        int ndims = 3;
        dims_     = dim3_t(nx, ny, nz);
        size_     = dims_.x * dims_.y * dims_.z;

        // allocate memory for the array
        buffer_ = new T [size_];
    }

    template <typename T>
    DArray<T>::~DArray() {
        if (buffer_) delete [] buffer_;
    }

    template <typename T>
    std::vector<Partition<T>> DArray<T>::create_partitions(int n_partitions) {
        dim3_t d     = dims_;
        int n_slices = dims_.x / n_partitions;
        int n_extra  = dims_.x % n_partitions;

        // vector to hold the partitions
        std::vector<Partition<T>> table;

        int offset = 0;
        for (int i = 0; i < n_partitions; i++) {
            if (i < n_extra) d.x = n_slices + 1;
            else
                d.x = n_slices;
            table.push_back(Partition<T>(d, slice(offset)));
            offset += d.x;
        }
        return table;
    }

    template <typename T>
    std::vector<Partition<T>> DArray<T>::create_partitions(int n_partitions, int halo) {
        int n_slices = dims_.x / n_partitions;
        int n_extra  = dims_.x % n_partitions;
  
        // vector to hold the partitions
        std::vector<Partition<T>> table;
        std::vector<int> locations;

        int offset = 0;
        locations.push_back(offset);
        for (int i = 0; i < n_partitions; i++) {
            if (i < n_extra)
                offset += n_slices + 1;
            else 
                offset += n_slices;
            locations.push_back(offset);
        }

        int h[2];
        for (int i = 0; i < n_partitions; i++) {
            int imin = std::max(locations[i] - halo, 0);
            if ( imin == 0 ) 
                h[0] = 0;
            else 
                h[0] = halo; 
            int imax = std::min(locations[i+1] + halo, dims_.x);
            if (imax == dims_.x) 
                h[1] = 0;
            else 
                h[1] = halo;
            dim3_t d(imax-imin, dims_.y, dims_.z);
            table.push_back(Partition<T>(d, slice(imin), h));
        } 
        return table;
    }

} // namespace tomocam
