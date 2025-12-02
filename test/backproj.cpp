
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "core/tomocam.h"
#include "io/hdf5/reader.h"
#include "io/hdf5/writer.h"
#include "memory/array_ops.h"
#include "memory/dist_array.h"
#include "utils/timer.h"

using json = nlohmann::json;

using tomocam::backproject;
using tomocam::DArray;
using tomocam::io::h5::Reader;
using tomocam::io::h5::Writer;
using tomocam::preprocessing::postproc;
using tomocam::preprocessing::preproc;
using tomocam::utils::Timer;

int main(int argc, char **argv) {

    // get JSON file
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <JSON file>" << std::endl;
        return 1;
    }

    // read JSON file
    std::ifstream json_file(argv[1]);
    if (!json_file.is_open()) {
        std::cerr << "Error: cannot open JSON file" << std::endl;
        return 1;
    }
    json cfg = json::parse(json_file);

    // get parameters
    std::string filename = cfg["filename"];
    std::string dataset = cfg["dataset"];
    std::string angles = cfg["angles"];
    std::string outfile = cfg["output"];
    int center = cfg["axis"];
    int ibeg = 0, iend = -1;
    // chcek for "slices" key
    if (cfg.find("slices") != cfg.end()) {
        auto slcs = cfg["slices"];
        ibeg = slcs[0];
        iend = slcs[1];
    }

    // read data
    Timer t;
    t.start();
    Reader h5fp(filename.c_str());
    auto sino = h5fp.read_sinogram<float>(dataset.c_str(), ibeg, iend);
    auto angs = h5fp.read<float>(angles.c_str());
    t.stop();
    std::cout << std::format("Data loading time: {:.3f} s\n", t.seconds());

    // if number of columns is even, drop one column
    float cen = static_cast<float>(center);

    // normalize sinogram
    auto maxv = tomocam::array::max(sino);
    auto minv = tomocam::array::min(sino);
    auto range = maxv - minv;
    auto sino2 = (sino - minv) / range;

    t.start();
    sino2 = preproc(sino2, cen);
    auto recn = backproject(sino2, angs, true);
    recn = postproc(recn, sino.ncols());
    t.stop();
    std::cout << std::format("Reconstruction time: {:.3f} s\n", t.seconds());
    Writer w(outfile.c_str());
    w.write("recon", recn);
    return 0;
}
