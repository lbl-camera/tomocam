#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "dist_array.h"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "machine.h"
#include "timer.h"
#include "tomocam.h"

int main(int argc, char **argv) {

    // read data json file
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <JSON file>" << std::endl;
        return 1;
    }

    // check if the file exists
    std::string json_file = argv[1];
    if (!std::filesystem::exists(json_file)) {
        std::cerr << "File not found: " << json_file << std::endl;
        return 1;
    }

    // read json file
    std::ifstream fp(json_file);
    json cfg = json::parse(fp);
    fp.close();

    // parameters
    std::string filename = cfg["filename"];
    std::string dataset = cfg["dataset"];
    int center = cfg["axis"];
    int ibeg = 0, iend = -1;
    if (cfg.find("slices") != cfg.end()) {
        auto slcs = cfg["slices"];
        ibeg = slcs[0];
        iend = slcs[1];
    }

    // read data
    tomocam::Timer read_timer;
    read_timer.start();
    tomocam::h5::H5Reader h5fp(filename.c_str());
    auto data = h5fp.read2<float>(dataset.c_str(), ibeg, iend);
    read_timer.stop();
    std::cout << "Data shape: " << data.nslices() << " x " << data.nrows()
              << " x " << data.ncols() << std::endl;
    std::cout << "Read time: " << read_timer.ms() << " ms" << std::endl;

    // if number of columns in odd, drop the last column
    data.ensure_odd_cols();

    // create angles
    std::vector<float> angs(data.ncols());
    float dtheta = M_PI / static_cast<float>(data.ncols());
    for (int i = 0; i < data.ncols(); i++) { angs[i] = i * dtheta; }

    // projection
    tomocam::Timer proj_timer;
    proj_timer.start();
    auto sino = tomocam::project<float>(data, angs, center);
    proj_timer.stop();

    // print time taken
    std::cout << "Projection time: " << proj_timer.ms() << " ms" << std::endl;

    // write hdf5 file
    const char *outfile = "sino.h5";
    tomocam::h5::H5Writer h5out(outfile);
    h5out.write<float>("sino", sino);
    return 0;
}
