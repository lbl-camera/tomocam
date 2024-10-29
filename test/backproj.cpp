
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
    tomocam::h5::Reader h5fp(filename.c_str());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto sino = h5fp.read_sinogram<float>(dataset.c_str(), ibeg, iend);
    auto angs = h5fp.read<float>(angles.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_read = t1 - t0;
    std::cout << "Elapsed time reading data: " << elapsed_read.count() << " s"
        << std::endl;

    // if number of columns is even, drop one column
    sino.dropcol();
    center -= 1;
    float cen = static_cast<float>(center);

    auto start = std::chrono::high_resolution_clock::now();
    auto sino2 = tomocam::preproc(sino, cen);
    auto temp = tomocam::backproject(sino2, angs, cen);
    auto sinoT = tomocam::postproc(temp, sino.ncols());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Backprojection time: " << elapsed.count() << " s"
        << std::endl;

    tomocam::h5::Writer w("backproj.h5");
    w.write("backproj", sinoT);
    return 0;
}
