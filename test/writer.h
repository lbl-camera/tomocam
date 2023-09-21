#include <H5Cpp.h>
#include <fstream>
#include <iostream>

#include "dist_array.h"

#ifndef TOMOCAM_WRITER__H
#define TOMOCAM_WRITER__H

namespace tomocam {

    class H5Writer {

      private:
        H5::H5File fp_;

      public:
        H5Writer(const char *filename): fp_(H5::H5File(filename, H5F_ACC_TRUNC)){}

        void write(const char *dname, DArray<float> &array) {

            // create dataspace
            auto d = array.dims();
            hsize_t dims[3];
            dims[0] = d.x;
            dims[1] = d.y;
            dims[2] = d.z; 
            H5::DataSpace spc(3, dims);
            H5::DataSet dset = fp_.createDataSet(dname, H5::PredType::NATIVE_FLOAT, spc);
            dset.write(array.data(), H5::PredType::NATIVE_FLOAT);
        }
        void write(const char *dname, DArray<double> &array) {

            // create dataspace
            auto d = array.dims();
            hsize_t dims[3];
            dims[0] = d.x;
            dims[1] = d.y;
            dims[2] = d.z; 
            H5::DataSpace spc(3, dims);
            H5::DataSet dset = fp_.createDataSet(dname, H5::PredType::NATIVE_DOUBLE, spc);
            dset.write(array.data(), H5::PredType::NATIVE_DOUBLE);
        }

    };
} // namespace tomocam
#endif // TOMOCAM_WRITER__H
