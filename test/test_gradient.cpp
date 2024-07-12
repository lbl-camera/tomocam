
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "dist_array.h"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "tomocam.h"

using json = nlohmann::json;

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
    int center = cfg["axis"];
    int ibeg = 0, iend = -1;
    // chcek for "slices" key
    if (cfg.find("slices") != cfg.end()) {
        auto slcs = cfg["slices"];
        ibeg = slcs[0];
        iend = slcs[1];
    }

    // read data
    tomocam::h5::H5Reader h5fp(filename.c_str());
    auto sino = h5fp.read_sinogram<float>(dataset.c_str(), ibeg, iend);
    auto angs = h5fp.read<float>(angles.c_str());

    // if number of columns is even, drop one column
    sino.ensure_odd_cols();

    // minus log
    sino.normalize();
    sino.minus_log();

    // allocate solution
    tomocam::dim3_t dims = {sino.nslices(), sino.ncols(), sino.ncols()};
    tomocam::DArray<float> x1(dims);
    x1.init(1.f);
    auto x2 = x1;

    // gradient 1
    auto y = tomocam::backproject(sino, angs, center);
    auto t1 = tomocam::project(x1, angs, center);
    auto t2 = tomocam::backproject(t1, angs, center);
    auto g1 = t2 - y;

    /*
    // gradient 2
    // create NUFFT grids
    std::vector<tomocam::NUFFT::Grid<float>> grids(4);
    for (int i = 0; i < 4; i++) {
        grids[i] = tomocam::NUFFT::Grid<float>(angs.size(), sino.ncols(),
            angs.data(), i);
    }
    auto tpl = tomocam::gradient(x2, y, grids);
    auto g2 = std::get<0>(tpl);
    auto f2 = std::get<1>(tpl);
    */
    // write to HDF5
    tomocam::h5::H5Writer h5fw("gradient.h5");
    h5fw.write("g1", y);
    h5fw.write("g2", g1);
    return 0;
}
