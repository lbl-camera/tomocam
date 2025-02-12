
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "dist_array.h"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "toeplitz.h"
#include "tomocam.h"

#include "timer.h"

int main(int argc, char **argv) {

    const int nprojs = 360;
    const int npixel = 511;
    const float center = static_cast<float>(npixel - 1) / 2;

    auto rng = NPRandom();

    // create data
    tomocam::DArray<float> sino(tomocam::dim3_t{1, nprojs, npixel});
    for (int i = 0; i < sino.size(); i++) {
        sino[i] = rng.rand<float>();
    }

    // create angles
    std::vector<float> angs(nprojs);
    for (int i = 0; i < nprojs; i++) {
        angs[i] = i * M_PI / nprojs;
    }

    // allocate solution array
    tomocam::dim3_t dims = {1, npixel, npixel};
    tomocam::DArray<float> x1(dims);
    x1.init(1.f);
    auto x2 = x1;

    // backproject sinogram
    auto yT = tomocam::backproject(sino, angs, center);

    // gradient 1
    auto t1 = tomocam::project(x1, angs, center) - sino;
    auto g1 = tomocam::backproject(t1, angs, center);

    // gradient 2
    // create NUFFT grids
    std::vector<tomocam::NUFFT::Grid<float>> nugrids(4);
    std::vector<tomocam::PointSpreadFunction<float>> psfs(4);
    for (int i = 0; i < 4; i++) {
        tomocam::NUFFT::Grid<float> grid(nprojs, npixel, angs.data(), i);
        nugrids[i] = grid;
        psfs[i] = tomocam::PointSpreadFunction<float>(grid);
    }

    // compute gradient
    float c = 0;
    auto g2 = tomocam::gradient2(x2, yT, psfs);

    // write to HDF5
    tomocam::h5::Writer h5fw("gradient.h5");
    h5fw.write("g1", g1);
    h5fw.write("g2", g2);

    // compare
    auto e = g1 - g2;
    std::cout << "Error: " << e.norm() / g1.norm() << std::endl;
    return 0;
}
