
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "dist_array.h"
#include "toeplitz.h"
#include "tomocam.h"

#include "timer.h"

int main(int argc, char **argv) {

    const int nslices = 256;
    const int nprojs = 360;
    const int npixel = 2047;
    const int center = npixel / 2;

    auto rng = NPRandom();

    // create data
    tomocam::DArray<double> sino(tomocam::dim3_t{nslices, nprojs, npixel});
    for (int i = 0; i < sino.size(); i++) {
        sino[i] = rng.rand<double>();
    }
    auto sino_norm = sino.norm();
    std::cout << "|| sino ||2 " << sino_norm << std::endl;

    // create angles
    std::vector<double> angs(nprojs);
    for (int i = 0; i < nprojs; i++) {
        angs[i] = i * M_PI / nprojs;
    }

    // allocate solution array
    tomocam::dim3_t dims = {nslices, npixel, npixel};
    tomocam::DArray<double> x1(dims);
    x1.init(1.f);
    auto x2 = x1;

    // error 1
    Timer timer01;
    timer01.start();
    auto t1 = tomocam::project<double>(x1, angs, center);
    auto t2 = t1 - sino;
    auto err1 = t2.norm();
    timer01.stop();

    // error 2
    // create NUFFT grids
    std::vector<tomocam::PointSpreadFunction<double>> psfs(4);
    for (int i = 0; i < 4; i++) {
        tomocam::NUFFT::Grid<double> grid(sino.nrows(), sino.ncols(),
            angs.data(), i);
        psfs[i] = tomocam::PointSpreadFunction<double>(grid);
    }

    // compute error
    auto y = tomocam::backproject<double>(sino, angs, center);
    Timer timer02;
    timer02.start();
    auto err2 = tomocam::function_value2(x2, y, psfs, sino_norm);
    timer02.stop();

    // compare
    std::cout << "Error 1: " << err1 << std::endl;
    std::cout << "Error 2: " << err2 << std::endl;
    std::cout << "Error2/Error1: " << err2 / err1 << std::endl;
    std::cout << "Time 1(ms): " << timer01.ms() << std::endl;
    std::cout << "Time 2(ms): " << timer02.ms() << std::endl;
    return 0;
}
