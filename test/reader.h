#include <H5Cpp.h>
#include <fstream>
#include <iostream>

#include "dist_array.h"

#ifndef TOMOCAM_READER__H
#define TOMOCAM_READER__H

namespace tomocam {
    class H5Reader {
      private:
        H5::H5File fp_;
        H5::DataSet dset_;
        H5::DataSpace fspace_;
        hsize_t dims_[3];

      public:
        H5Reader(const char *filename): 
            fp_(H5::H5File(filename, H5F_ACC_RDONLY)), dims_{ 0, 0, 0 } {}

        void setDataset(const char *dataset) {
            dset_ = fp_.openDataSet(dataset);
            fspace_ = dset_.getSpace();
            int ndims = fspace_.getSimpleExtentDims(dims_, NULL);
        }

        DArray<float> read_sinogram(int slices, int begin = 0) {

            // return value
            DArray<float> a(dim3_t(slices, dims_[0], dims_[2]));
        
            // create memspace
            hsize_t out_dims[3] = {1, dims_[0], dims_[2]};
            H5::DataSpace memspace(3, out_dims);

            // select a slice at a time
            for (int i = 0; i < slices; i++) {
                hsize_t offset[3] = {0, (unsigned long)(begin + i), 0};
                hsize_t count[3] = {dims_[0], 1, dims_[2]};
                fspace_.selectHyperslab(H5S_SELECT_SET, count, offset);

                void *ptr = (void *)a.slice(i);
                dset_.read(ptr, H5::PredType::NATIVE_FLOAT, memspace, fspace_);
            }
            return a;
        }

        std::vector<float> read_angles(const char * name) {
            auto ds = fp_.openDataSet(name);
            auto fs = ds.getSpace();
            hsize_t dims[3]  = { 0, 0, 0 }; 
            int ndim = fs.getSimpleExtentDims(dims);
            size_t size = 1;
			for (int i = 0; i < ndim; i++) size *=  dims[i];
            H5::DataSpace ms(ndim, dims);
            
            std::vector<float> a(size, 0);
            ds.read(a.data(), H5::PredType::NATIVE_FLOAT, ms, fs);
            return a;
        }
    };
} // namespace tomocam
#endif // TOMOCAM_READER__H
