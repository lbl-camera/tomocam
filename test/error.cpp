
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

    const int nslices = 16;
    const int nprojs = 360;
    const int npixel = 2047;
    const int center = npixel / 2;

    auto rng = NPRandom();

    // create data
    tomocam::DArray<float> sino(tomocam::dim3_t{nslices, nprojs, npixel});
    for (int i = 0; i < sino.size(); i++) {
        sino[i] = rng.rand<float>();
    }
    auto sino_norm = sino.norm();
    std::cout << "|| sino ||\x00b2 " << sino_norm << std::endl;

    // create angles
    std::vector<float> angs(nprojs);
    for (int i = 0; i < nprojs; i++) {
        angs[i] = i * M_PI / nprojs;
    }

    // allocate solution array
    tomocam::dim3_t dims = {nslices, npixel, npixel};
    tomocam::DArray<float> x1(dims);
    x1.init(1.f);
    auto x2 = x1;

    // error 1
    Timer time1;
    time1.start();
    auto t1 = tomocam::project<float>(x1, angs);
    auto t2 = t1 - sino;
    auto err1 = t2.norm();
    time1.stop();

    // error 2
    // create NUFFT grids
    std::vector<tomocam::NUFFT::Grid<float>> grids(4);
    std::vector<tomocam::PointSpreadFunction<float>> psfs(4);
    for (int i = 0; i < 4; i++) {
        tomocam::NUFFT::Grid<float> grid(sino.nrows(), sino.ncols(),
            angs.data(), i);
        grids[i] = grid;
        psfs[i] = tomocam::PointSpreadFunction<float>(grid);
        psfs[i].create_plans(4);
    }

    // error 2
    Timer time2;
    time2.start();
    auto err2 = tomocam::function_value(x2, sino, grids);
    time2.stop();

    // compute error
    auto y = tomocam::backproject<float>(sino, angs, false);
    Timer time3;
    time3.start();
    auto err3 = tomocam::function_value2(x2, y, psfs, sino_norm);
    time3.stop();

    // compare
    std::cout << "Error 1: " << err1 << std::endl;
    std::cout << "Error 2: " << err2 << std::endl;
    std::cout << "Error 3: " << err3 << std::endl;
    std::cout << "Error2/Error1: " << err2 / err1 << std::endl;
    std::cout << "Error3/Error1: " << err3 / err1 << std::endl;
    std::cout << "Time 1: " << time1.ms() << " ms" << std::endl;
    std::cout << "Time 2: " << time2.ms() << " ms" << std::endl;
    std::cout << "Time 3: " << time3.ms() << " ms" << std::endl;
    return 0;
}
