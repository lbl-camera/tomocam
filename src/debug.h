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

#include "dist_array.h"
#include "dev_array.h"
#include "hdf5/writer.h"

#pragma once

namespace tomocam {

    template <typename T>
    class HostArray {
        private:
            dim3_t dims_;
            std::vector<T> data_;

        public:
            HostArray(const DeviceArray<T> &d) : dims_(d.dims()) {
                data_ = d.copy_to_host();
            }

            T operator()(int i, int j, int k) {
                return data_[i * dims_.y * dims_.z + j * dims_.z + k];
            }

            int nslice() const { return dims_.x; }
            int nrows() const { return dims_.y; }
            int ncols() const { return dims_.z; }
    };

    template <typename T>
    inline void write_h5(const DeviceArray<T> &arr) {
        h5::Writer writer("debug.h5");

        // copy data to DAarray
        DArray<T> data(arr.dims());
        SAFE_CALL(cudaMemcpy(data.begin(), arr.dev_ptr(),
            arr.size() * sizeof(T), cudaMemcpyDeviceToHost));
        writer.write<T>("data", data);
    }

    template <typename T>
    concept Real = std::floating_point<T>;

    template <typename T>
    concept Complex_t = requires(T t) {
        { t.real() }
        -> Real;
        { t.imag() }
        -> Real;
    };

    template <typename Real>
    void printf(const DeviceArray<Real> &d) {
        HostArray<Real> h(d);

        int nrows = h.nrows();
        if (nrows > 5) nrows = 5;
        int ncols = h.ncols();
        if (ncols > 5) ncols = 5;
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < h.ncols(); j++) {
                std::cout << h(0, i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    template <typename Complex_t>
    void print(const DeviceArray<Complex_t> &d) {
        HostArray<Complex_t> h(d);

        int nrows = h.nrows();
        if (nrows > 5) nrows = 5;
        int ncols = h.ncols();
        if (ncols > 5) ncols = 5;
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < h.ncols(); j++) {
                std::cout << "(" << h(0, i, j).real() << ", "
                    << h(0, i, j).imag() << ") ";
            }
            std::cout << std::endl;
        }
    }

} // namespace tomocam
