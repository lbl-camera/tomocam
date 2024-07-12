
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dev_array.h"
#include "dist_array.h"
#include "fft.h"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "internals.h"
#include "nufft.h"
#include "timer.h"
#include "tomocam.h"

using json = nlohmann::json;

int main(int argc, char **argv) {

    // read json file
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <JSON file>" << std::endl;
        return 1;
    }
    std::ifstream json_file(argv[1]);
    json cfg = json::parse(json_file);
    std::string filename = cfg["filename"];
    std::string dataset = cfg["dataset"];
    std::string angles = cfg["angles"];

    // read data
    tomocam::h5::H5Reader fp(filename.c_str());
    auto sino = fp.read_sinogram<float>(dataset.c_str());
    auto angs = fp.read<float>(angles.c_str());
    float * theta = angs.data();

    // make sure the last dimension is odd
    sino.ensure_odd_cols();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2 = {d1.x, d1.z, d1.z};
    tomocam::DArray<float> recn(d2);

    // center of rotation
    int center = sino.ncols() / 2;
    int cen_shift = 0;

    // nufft grid
    int ncols = d1.z + 2 * cen_shift;
    int nproj = d1.y;
    tomocam::NUFFT::Grid grid(nproj, ncols, theta, 0);

    // create partitions
    auto part1 = tomocam::create_partitions(sino, 1)[0];
    auto part2 = tomocam::create_partitions(recn, 1)[0];

    // move data to GPU RAM
    tomocam::DeviceArray<float> d_sino(part1);
    auto d_recn = tomocam::backproject(d_sino, grid, cen_shift, 0);
    d_recn.copy_to(part2, 0);

    tomocam::h5::H5Writer w("backproj.h5");
    w.write("backproj", recn);
    return 0;
}
