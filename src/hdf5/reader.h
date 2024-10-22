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
        class Reader {
          private:
            hid_t fp_;

          public:
            Reader(const char *filename) {
                fp_ = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
            }

            ~Reader() { H5Fclose(fp_); }

            // get data dimenstions
            int dims(const char *dsetname, int dim) {

                hid_t dset = H5Dopen2(fp_, dsetname, H5P_DEFAULT);
                hid_t dspc = H5Dget_space(dset);
                hsize_t dims[3] = {0, 0, 0}; // max 3D
                int ndim = H5Sget_simple_extent_dims(dspc, dims, NULL);
                if (ndim != 3) { throw std::runtime_error("Data is not 3D"); }
                if (dim < 0 || dim > 2) {
                    throw std::runtime_error("Invalid dimension");
                }
                H5Sclose(dspc);
                H5Dclose(dset);
                return static_cast<int>(dims[dim]);
            }

            /** read projection data into sinogram format
             * @param dataset dataset name
             * @param nslice number of slices to read, 0 for all
             * @param begin starting slice index, 0 is default
             * @return sinogram data
             */
            template <typename T>
            DArray<T> read_sinogram(const char *dataset, hsize_t begin = 0,
                hsize_t end = -1) {

                // open dataset
                hid_t dset = H5Dopen2(fp_, dataset, H5P_DEFAULT);

                // get dataspace
                hid_t fspace = H5Dget_space(dset);

                // get full data dimensions
                hsize_t dims[3] = {0, 0, 0};
                int ndim = H5Sget_simple_extent_dims(fspace, dims, NULL);

                if (end == -1) { end = dims[1]; }

                // check bounds
                if ((end - begin) > dims[1]) {
                    throw std::runtime_error("Index out of bounds");
                }
                hsize_t nslice = end - begin;

                // data type
                hid_t dtype = H5Dget_type(dset);
                if (!(H5Tequal(dtype, getH5Dtype<T>()))) {
                    throw std::runtime_error("Data type mismatch");
                }


                // create memory space for reading
                hsize_t out_dims[3] = {dims[0], nslice, dims[2]};
                hid_t out_space = H5Screate_simple(3, out_dims, NULL);
                DArray<T> A({(int)dims[0], (int)nslice, (int)dims[2]});

                // hyperslab selection
                hsize_t count[3] = {dims[0], nslice, dims[2]};
                hsize_t start[3] = {0, begin, 0};
                H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count,
                    NULL);
                H5Dread(dset, dtype, out_space, fspace, H5P_DEFAULT, A.begin());

                // allocate return value
                DArray<T> B({(int)nslice, (int)dims[0], (int)dims[2]});

                // transpose data
                #pragma omp parallel for
                for (uint64_t i = 0; i < nslice; i++)
                    for (uint64_t j = 0; j < dims[0]; j++)
                        for (uint64_t k = 0; k < dims[2]; k++)
                            B(i, j, k) = A(j, i, k);

                // clean up
                H5Sclose(out_space);
                H5Sclose(fspace);
                H5Dclose(dset);
                return B;
            }

            template <typename T>
            DArray<T> read2(const char *dataset, int begin = 0, int end = -1) {

                // open dataset
                hid_t dset = H5Dopen2(fp_, dataset, H5P_DEFAULT);

                // get dataspace
                hid_t fspace = H5Dget_space(dset);

                // get full data dimensions
                hsize_t dims[3] = {0, 0, 0};
                int ndim = H5Sget_simple_extent_dims(fspace, dims, NULL);
                if (end == -1) { end = dims[0]; }

                // check bounds
                if ((end - begin) > dims[0]) {
                    throw std::runtime_error("Index out of bounds");
                }
                hsize_t nslice = end - begin;

                // data type
                hid_t dtype = H5Dget_type(dset);
                if (!(H5Tequal(dtype, getH5Dtype<T>()))) {
                    throw std::runtime_error("Data type mismatch");
                }

                // create memory space for reading
                hsize_t out_dims[3] = {nslice, dims[1], dims[2]};
                hid_t out_space = H5Screate_simple(3, out_dims, NULL);
                DArray<T> A({(int)nslice, (int)dims[1], (int)dims[2]});

                // hyperslab selection
                hsize_t count[3] = {nslice, dims[1], dims[2]};
                hsize_t start[3] = {(hsize_t)begin, 0, 0};
                H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count,
                    NULL);
                H5Dread(dset, dtype, out_space, fspace, H5P_DEFAULT, A.begin());

                // clean up
                H5Sclose(out_space);
                H5Sclose(fspace);
                H5Dclose(dset);
                return A;
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
