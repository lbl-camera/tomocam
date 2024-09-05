
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


int main(int argc, char **argv) {

    const int nprojs = 5;
    const int npixel = 5;
    const int center = npixel / 2;

    auto random = []() { return static_cast<float>(rand()) / RAND_MAX; };

    // create data
    tomocam::DArray<float> sino(tomocam::dim3_t{1, nprojs, npixel});
    for (int i = 0; i < sino.size(); i++) {
        sino[i] = random();
        // sino[i] = 0;
    }

    // create angles
    std::vector<float> angs(nprojs);
    for (int i = 0; i < nprojs; i++) { angs[i] = i * M_PI / nprojs; }

    // allocate solution array
    tomocam::dim3_t dims = {1, npixel, npixel};
    tomocam::DArray<float> x1(dims);
    x1.init(1.f);
    auto x2 = x1;

    // gradient 1
    auto t1 = tomocam::project(x1, angs, center);
    auto g1 = tomocam::backproject(t1, angs, center);

    for (int i = 0; i < npixel; i++) {
        for (int j = 0; j < npixel; j++) { std::cout << g1(0, i, j) << " "; }
        std::cout << std::endl;
    }
    std::cout << "----------------" << std::endl;

    // gradient 2
    // create NUFFT grids
    std::vector<tomocam::PointSpreadFunction<float>> psfs(4);
    for (int i = 0; i < 4; i++) {
        tomocam::NUFFT::Grid<float> grid(sino.nrows(), sino.ncols(),
            angs.data(), i);
        psfs[i] = tomocam::PointSpreadFunction<float>(grid);
    }

    // compute gradient
    auto y = tomocam::backproject(sino, angs, center);
    auto g2 = tomocam::gradient2(x2, y, psfs);

    for (int i = 0; i < npixel; i++) {
        for (int j = 0; j < npixel; j++) { std::cout << g1(0, i, j) << " "; }
        std::cout << std::endl;
    }
    std::cout << "----------------" << std::endl;
    exit(1);

    // write to HDF5
    tomocam::h5::Writer h5fw("gradient.h5");
    h5fw.write("g1", t1);
    h5fw.write("g2", g2);

    // compare
    auto e = g1 - g2;
    std::cout << "Error: " << e.norm() << std::endl;
    return 0;
}
