
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "hdf5/writer.h"

using json = nlohmann::json;

int main(int argc, char **argv) {

    // create a test data
    tomocam::dim3_t dims = {4, 20, 48};
    tomocam::DArray<float> data(dims);
    for (uint64_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<float>(i);
    }

    // create a copy of the data
    auto data2 = data;
    auto start = std::chrono::high_resolution_clock::now();
    data2.ensure_odd_cols();
    auto dt = std::chrono::high_resolution_clock::now() - start;

    // write data to hdf5
    tomocam::h5::H5Writer writer("test_dropcol.h5");
    writer.write("data", data);
    writer.write("data2", data2);

    // print data shape
    std::cout
        << "Data2 shape: (" << data2.nslices() << ", " << data2.nrows() << ", "
        << data2.ncols() << "), time: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(dt).count()
        << " ms" << std::endl;
    return 0;
}
