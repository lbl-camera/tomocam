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

#include <vector>

namespace tomocam {

    template <typename T>
    DArray<T>::DArray(dim3_t dim) {
        dims_     = dim;
        size_     = dims_.x * dims_.y * static_cast<uint64_t>(dims_.z);

        // allocate memory for the array
        buffer_ = new T [size_];
    }

    /* destructor */
    template <typename T>
    DArray<T>::~DArray() {
        if (buffer_) delete [] buffer_;
    }

    /* copy constructor */
    template <typename T>
    DArray<T>::DArray(const DArray<T> &other) {
        if (this == &other) return;

        dims_ = other.dims_;
        size_ = other.size_;
        buffer_ = new T [size_];
        std::copy(other.buffer_, other.buffer_ + size_, buffer_);
    }

    /* assignment operator */
    template <typename T>
    DArray<T>& DArray<T>::operator=(const DArray<T> & other) {
        if (this == &other) return *this;
        if (size_ != other.size_) throw std::runtime_error("error: size mismatch!");
        std::copy(other.buffer_, other.buffer_ + size_, buffer_);
        return *this;
    }

    /* move constructor */
    template <typename T>
    DArray<T>::DArray(DArray<T> &&other) {
        if (this == &other) return;

        dims_ = other.dims_;
        size_ = other.size_;
        buffer_ = other.buffer_;
        other.buffer_ = nullptr;
    }

    /* move assignment */
    template <typename T>
    DArray<T> &DArray<T>::operator=(DArray<T> &&other) {
        if (this == &other) return *this;

        if (buffer_) delete [] buffer_;
        dims_ = other.dims_;
        size_ = other.size_;
        buffer_ = other.buffer_;
        other.buffer_ = nullptr;
        return *this;
    }

    // addition operators
    template <typename T>
    DArray<T> DArray<T>::operator+(const DArray<T> & rhs) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] + rhs.buffer_[i];
        return rv;
    }
    template <typename T>
    DArray<T> DArray<T>::operator+(T rhs) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] + rhs;
        return rv;
    }
     
    // subtraction operators
    template <typename T>
    DArray<T> DArray<T>::operator-(const DArray<T> & rhs) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] - rhs.buffer_[i];
        return rv;
    }
    template <typename T>
    DArray<T> DArray<T>::operator-(T rhs) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] - rhs;
        return rv;
    }

    // add-assign
    template <typename T>
    DArray<T> & DArray<T>::operator+=(const DArray<T> &rhs) {
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            buffer_[i] += rhs.buffer_[i];
        return *this;
    }

    // multiplication
    template <typename T>
    DArray<T> DArray<T>::operator*(const DArray<T> &rhs) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] * rhs.buffer_[i];
        return rv;
    }

    // division
    template <typename T>
    DArray<T> DArray<T>::operator/(const DArray<T> &rhs) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] / rhs.buffer_[i];
        return rv;
    }

    // scaling
    template <typename T>
    DArray<T> DArray<T>::operator*(T m) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] * m;
        return rv;
    }
    template <typename T>
    DArray<T> DArray<T>::operator/(T m) {
        DArray<T> rv(dims_);
        #pragma omp parallel for
        for (uint64_t i = 0; i < size_; i++)
            rv.buffer_[i] = buffer_[i] / m;
        return rv;
    }

    /* subdivide array into N partitions */
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

    /* subdivide array into N partitions, with n halo layers on boundaries */
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
            if (i == 0) 
                h[0] = 0;
            else 
                h[0] = halo; 
            int imax = std::min(locations[i+1] + halo, dims_.x);
            if (i == n_partitions-1)
                h[1] = 0;
            else 
                h[1] = halo;
            dim3_t d(imax-imin, dims_.y, dims_.z);
            table.push_back(Partition<T>(d, slice(imin), h));
        } 
        return table;
    }

} // namespace tomocam
