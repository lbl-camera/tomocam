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
    std::vector<Partition<T>> Partition<T>::sub_partitions(int partition_size, int halo) {
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
        dims_     = dim;
        size_     = dims_.x * dims_.y * static_cast<uint64_t>(dims_.z);
        owns_buffer_ = true;

        // allocate memory for the array
        buffer_ = new T [size_];
    }

    // for calling from python
    template <typename T>
    DArray<T>::DArray(np_array_t<T> np_array) {
        int ndims = np_array.ndim();
        if (ndims == 1) 
            dims_ = { 1, 1, (int)np_array.shape(0) };
        else if (ndims == 2) 
            dims_ = { 1, (int)np_array.shape(0), (int)np_array.shape(1) };
        else if (ndims == 3)
            dims_ = { (int)np_array.shape(0), (int)np_array.shape(1), (int)np_array.shape(2) };
        else 
            throw std::runtime_error("Unsupported numpy array.");
        size_ = dims_.x * dims_.y * static_cast<uint64_t>(dims_.z);
        buffer_ = (T *) np_array.request().ptr;
        owns_buffer_ = false;
    }
 
    /* destructor */
    template <typename T>
    DArray<T>::~DArray() {
        if (owns_buffer_) delete [] buffer_;
    }

    /* copy constructor */
    template <typename T>
    DArray<T>::DArray(const DArray<T> &other) {
        if (this == &other) return;
        dims_ = other.dims_;
        size_ = other.size_;
        buffer_ = new T [size_];
        std::copy(other.buffer_, other.buffer_ + size_, buffer_);
        owns_buffer_ = true;
    }

    /* assignment operator */
    template <typename T>
    DArray<T>& DArray<T>::operator=(const DArray<T> & other) {
        if (this == &other) return *this;
        if (dims_ != other.dims_) throw std::runtime_error("size mismatch!");
        std::copy(other.buffer_, other.buffer_ + size_, buffer_);
        return *this;
    }


    /* move constructor */
    template <typename T>
    DArray<T>::DArray(DArray<T> &&other) {
        dims_ = other.dims_;
        size_ = other.size_;
        owns_buffer_ = other.owns_buffer_;
        buffer_ = other.buffer_;

        other.dims_ = dim3_t{0, 0, 0};
        other.size_ = 0;
        other.buffer_ = nullptr;
    }

    /* move assignment */
    template <typename T>
    DArray<T> &DArray<T>::operator=(DArray<T> &&other) {
        dims_ = other.dims_;
        size_ = other.size_;
        owns_buffer_ = other.owns_buffer_;
        buffer_ = other.buffer_;

        other.dims_ = dim3_t{0, 0, 0};
        other.size_ = 0;
        other.buffer_ = nullptr;
        return *this;
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
