#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "core/internals.h"
#include "core/tomocam.h"
#include "gpu/padding.cuh"
#include "io/hdf5/reader.h"
#include "io/hdf5/writer.h"
#include "memory/dev_array.h"
#include "memory/dist_array.h"

using tomocam::backproject;
using tomocam::io::h5::Reader;
using tomocam::io::h5::Writer;
using tomocam::preprocessing::postproc;
using tomocam::preprocessing::preproc;

uint64_t millisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
               high_resolution_clock::now().time_since_epoch())
        .count();
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
    Reader reader(fname.c_str());
    auto sino = reader.read_sinogram<float>(dataset.c_str(), 0, 1);
    auto angles = reader.read<float>(angs.c_str());

    // hdf5 file
    Writer fp("padding_test.h5");

    // write sinogram to file
    fp.write("unpadded", sino);

    // pad sinogram
    auto sino2 = preproc(sino, center);
    fp.write("padded", sino2);

    auto recon = backproject(sino2, angles);
    fp.write("backproj", recon);

    auto recon2 = postproc(recon, sino.ncols());
    fp.write("cropped", recon2);

    return 0;
}
