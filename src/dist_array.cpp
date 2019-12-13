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

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

namespace tomocam {
    template <typename T>
    DArray<T>::DArray(dim3_t dim) {
        // limit ndims to 3
        int ndims = 3;
        dims_     = dim;
        size_     = dims_.x * dims_.y * dims_.z;

        // allocate memory for the array
        size_t buffersize = sizeof(T) * size_;
        cudaMallocHost((void **)&buffer_, buffersize);
    }

    // for calling from python
    template<typename T>
    DArray<T>::DArray(int nx, int ny, int nz) {
        // limit ndims to 3
        int ndims = 3;
        dims_     = dim3_t(nx, ny, nz);
        size_     = dims_.x * dims_.y * dims_.z;

        // allocate memory for the array
        size_t buffersize = sizeof(T) * size_;
        cudaMallocHost((void **)&buffer_, buffersize);
    }

    template <typename T>
    DArray<T>::~DArray() {
        std::cerr << "Destroying DAarray ... " << std::endl;
        cudaDeviceSynchronize();
        if (buffer_) cudaFree(buffer_);
    }

    template <typename T>
    std::vector<Partition<T>> DArray<T>::create_partitions(int n_partitions) {
        dim3_t d        = dims_;
        int    n_slices = dims_.x / n_partitions;
        int    n_extra  = dims_.x % n_partitions;

        // vector to hold the partitions
        std::vector<Partition<T>> table;

        int offset = 0;
        for (int i = 0; i < n_partitions; i++) {
            if (i < n_extra)
                d.x = n_slices + 1;
            else
                d.x = n_slices;
            table.push_back(Partition<T>(d, slice(offset)));
            offset += d.x;
        }
        return table;
    }

    template <typename T>
    T &DArray<T>::operator()(int i) {
        return buffer_[i];
    }

    template <typename T>
    T &DArray<T>::operator()(int i, int j) {
        return buffer_[idx_(i, j)];
    }

    template <typename T>
    T &DArray<T>::operator()(int i, int j, int k) {
        return buffer_[idx_(i, j, k)];
    }
} // namespace tomocam
