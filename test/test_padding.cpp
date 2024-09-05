#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "dev_array.h"
#include "dist_array.h"
#include "gpu/padding.cuh"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "internals.h"
#include "tomocam.h"

uint64_t millisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {

    // read JSON configuration file
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <JSON file>" << std::endl;
        return 1;
    }

    std::ifstream ifs(argv[1]);
    auto config = json::parse(ifs);

    // file name
    std::string fname = config["filename"];
    std::string dataset = config["dataset"];
    int center = config["axis"];

    // read hdf5 file
    tomocam::h5::Reader reader(fname.c_str());
    auto sino = reader.read_sinogram<float>(dataset.c_str(), 1, 0);

    // hdf5 file
    tomocam::h5::Writer fp("padding_test.h5");
    // write sinogram to file
    fp.write("unpadded", sino);

    // create a device array
    tomocam::DeviceArray<float> d_sino(sino.dims());
    SAFE_CALL(cudaMemcpy(d_sino.dev_ptr(), sino.begin(),
        sizeof(float) * sino.size(), cudaMemcpyHostToDevice));

    int shift = center - d_sino.ncols() / 2;
    std::cout << "Shift: " << shift << std::endl;
    tomocam::PadType type = tomocam::PadType::RIGHT;
    if (shift < 0) { type = tomocam::PadType::LEFT; }
    auto padded = tomocam::gpu::pad1d(d_sino, shift, type);

    // copy to host
    tomocam::DArray<float> arr(padded.dims());
    SAFE_CALL(cudaMemcpy(arr.begin(), padded.dev_ptr(),
        sizeof(float) * padded.size(), cudaMemcpyDeviceToHost));
    fp.write("padded", arr);

    return 0;
}
