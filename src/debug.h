

#include "dist_array.h"
#include "dev_array.h"
#include "hdf5/writer.h"


#pragma once

namespace tomocam {

    template <typename T>
    inline void write_h5(const DeviceArray<T> &arr) {
        h5::H5Writer writer("test.h5");

        // copy data to DAarray
        DArray<T> data(arr.dims());
        SAFE_CALL(cudaMemcpy(data.begin(), arr.dev_ptr(),
            arr.size() * sizeof(T), cudaMemcpyDeviceToHost));
        writer.write<T>("data", data);
    }

} // namespace tomocam

