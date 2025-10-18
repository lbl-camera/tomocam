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
#include "utils.h"
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
    std::string angs = config["angles"];
    float center = config["axis"];

    // read hdf5 file
    tomocam::h5::Reader reader(fname.c_str());
    auto sino = reader.read_sinogram<float>(dataset.c_str(), 0, 1);
    auto angles = reader.read<float>(angs.c_str());

    // hdf5 file
    tomocam::h5::Writer fp("padding_test.h5");

    // write sinogram to file
    fp.write("unpadded", sino);

    // pad sinogram
    auto sino2 = tomocam::preproc(sino, center);
    fp.write("padded", sino2);

    auto recon = tomocam::backproject(sino2, angles);
    fp.write("backproj", recon);

    auto recon2 = tomocam::postproc(recon, sino.ncols());
    fp.write("cropped", recon2);

    return 0;
}
