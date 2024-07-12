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
#include <concepts>
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <type_traits>

#include "hdf5/h5dtype.h"

#ifndef TOMOCAM_WRITER__H
#define TOMOCAM_WRITER__H

namespace tomocam {

    template <typename T>
    concept Complex = requires(T a) {
        a.real();
        a.imag();
    };

    namespace h5 {
        class H5Writer {
          private:
            hid_t file_;

          public:
            H5Writer(const char *filename) {
                file_ = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT,
                    H5P_DEFAULT);
            }

            ~H5Writer() { H5Fclose(file_); }

            template <typename T>
            void write(const char *dataset_name, const DArray<T> &array) {
                hsize_t dims[3];
                dims[0] = array.nslices();
                dims[1] = array.nrows();
                dims[2] = array.ncols();

                hid_t space = H5Screate_simple(3, dims, NULL);
                auto dtype = getH5Dtype<T>();
                hid_t dset = H5Dcreate(file_, dataset_name, dtype, space,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    array.begin());

                // Close resources
                H5Dclose(dset);
                H5Sclose(space);
            }

            template <Complex T>
            void write(const char *dataset_name, const DArray<T> &array) {
                hsize_t dims[3];
                dims[0] = array.nslices();
                dims[1] = array.nrows();
                dims[2] = array.ncols();

                hid_t space = H5Screate_simple(3, dims, NULL);
                auto dtype = makeComplexType<T>();
                hid_t dset = H5Dcreate(file_, dataset_name, dtype, space,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    array.begin());

                // Close resources
                H5Dclose(dset);
                H5Sclose(space);
            }

            template <typename T>
            void write(const char *dataset_name, const std::vector<T> &array) {
                hsize_t dims[1];
                dims[0] = array.size();

                hid_t space = H5Screate_simple(1, dims, NULL);
                auto dtype = getH5Dtype<T>();
                hid_t dset = H5Dcreate(file_, dataset_name, dtype, space,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    array.data());

                // Close resources
                H5Dclose(dset);
                H5Sclose(space);
            }

            template <Complex T>
            void write(const char *dataset_name, const std::vector<T> &array) {
                hsize_t dims[1];
                dims[0] = array.size();

                hid_t space = H5Screate_simple(1, dims, NULL);
                auto dtype = makeComplexType<T>();
                hid_t dset = H5Dcreate(file_, dataset_name, dtype, space,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    array.data());

                // Close resources
                H5Dclose(dset);
                H5Sclose(space);
            }
        };
    } // namespace h5
} // namespace tomocam
#endif // TOMOCAM_WRITER__H
