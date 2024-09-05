
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "dist_array.h"
#include "toeplitz.h"
#include "tomocam.h"


int main(int argc, char **argv) {

    const int nprojs = 5;
    int npixel = 5;
    if (argc > 1) { npixel = std::stoi(argv[1]); }
    const int center = npixel / 2;

    auto random = []() { return static_cast<double>(rand()) / RAND_MAX; };

    // create data
    tomocam::DArray<double> sino(tomocam::dim3_t{1, nprojs, npixel});
    for (int i = 0; i < sino.size(); i++) {
        sino[i] = random();
    }
    auto sino_norm = sino.norm();

    // create angles
    std::vector<double> angs(nprojs);
    for (int i = 0; i < nprojs; i++) { angs[i] = i * M_PI / nprojs; }

    // allocate solution array
    tomocam::dim3_t dims = {1, npixel, npixel};
    tomocam::DArray<double> x1(dims);
    x1.init(1.f);
    auto x2 = x1;

    // error 1
    auto t1 = tomocam::project(x1, angs, center);
    auto t2 = t1 - sino;
    auto err1 = t2.norm();

    // error 2
    // create NUFFT grids
    std::vector<tomocam::PointSpreadFunction<double>> psfs(4);
    for (int i = 0; i < 4; i++) {
        tomocam::NUFFT::Grid<double> grid(sino.nrows(), sino.ncols(),
            angs.data(), i);
        psfs[i] = tomocam::PointSpreadFunction<double>(grid);
    }

    // compute error
    auto y = tomocam::backproject(sino, angs, center);
    auto err2 = tomocam::function_value(x2, y, psfs, sino_norm);

    // compare
    std::cout << "Error 1: " << err1 << std::endl;
    std::cout << "Error 2: " << err2 << std::endl;
    std::cout << "Error2/Error1: " << err2 / err1 << std::endl;
    return 0;
}
