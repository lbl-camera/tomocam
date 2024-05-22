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

#include <fstream>
#include <hdf5.h>
#include <iostream>

#include "dist_array.h"
#include "hdf5/h5dtype.h"

#ifndef TOMOCAM_READER__H
#    define TOMOCAM_READER__H

namespace tomocam {
    namespace h5 {
        class H5Reader {
          private:
            hid_t fp_;

          public:
            H5Reader(const char *filename) {
                fp_ = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
            }

            ~H5Reader() { H5Fclose(fp_); }

            // read projection data into sinogram format
            template <typename T>
            DArray<T> read_sinogram(const char *dataset, hsize_t nslice,
                hsize_t begin = 0) {

                // open dataset
                hid_t dset = H5Dopen2(fp_, dataset, H5P_DEFAULT);

                // get dataspace
                hid_t fspace = H5Dget_space(dset);

                // get full data dimensions
                hsize_t dims[3] = {0, 0, 0};
                int ndim = H5Sget_simple_extent_dims(fspace, dims, NULL);

                // check bounds
                if (begin + nslice > dims[1]) {
                    throw std::runtime_error("Index out of bounds");
                }

                // data type
                hid_t dtype = H5Dget_type(dset);
                if (!(H5Tequal(dtype, getH5Dtype<T>()))) {
                    throw std::runtime_error("Data type mismatch");
                }

                // allocate return value
                dim3_t shape = {(int)nslice, (int)dims[0], (int)dims[2]};
                DArray<T> a(shape);

                // read sinogram
                /* projection data is strored in the order of [angle, h, r]
                 * we need to transpose the data to [h, angle, r]. This means
                 * reading one slice at a time with appropriate offset.
                 **/

                // create memory space for reading
                hsize_t out_dims[3] = {1, dims[0], dims[2]};
                hid_t out_space = H5Screate_simple(3, out_dims, NULL);

                // hyperslab selection
                hsize_t count[3] = {dims[0], 1, dims[2]};
                for (hsize_t i = begin; i < begin + nslice; i++) {
                    hsize_t start[3] = {0, i, 0};
                    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL,
                        count, NULL);
                    H5Dread(dset, dtype, out_space, fspace, H5P_DEFAULT,
                        a.slice(i - begin));
                }
                return a;
            }

            template <typename T>
            std::vector<T> read(const char *dataset) {
                // open dataset
                hid_t dset = H5Dopen2(fp_, dataset, H5P_DEFAULT);

                // get dataspace
                hid_t fspace = H5Dget_space(dset);

                // get full data dimensions
                hsize_t dims[1] = {0};
                int ndim = H5Sget_simple_extent_dims(fspace, dims, NULL);

                // data type
                hid_t dtype = getH5Dtype<T>();
                if (!(H5Tequal(dtype, getH5Dtype<T>()))) {
                    throw std::runtime_error("Data type mismatch");
                }

                // allocate return value
                std::vector<T> a(dims[0]);

                // read sinogram
                H5Dread(dset, dtype, H5S_ALL, fspace, H5P_DEFAULT, a.data());
                return a;
            }
        };
    } // namespace h5
} // namespace tomocam
#endif // TOMOCAM_READER__H
