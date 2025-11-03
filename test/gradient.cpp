
#include <filesystem>
#include <fstream>
#include <iostream>
#include <format>
#include <nlohmann/json.hpp>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "toeplitz.h"
#include "tomocam.h"
//#include "timer.h"

using json = nlohmann::json;
int main(int argc, char **argv) {


    // define size of the reconstruction
    int nslices = 16;
    int nrows = 2047;
    int ncols = 2047;

    // allocate solution array
    int nprojs = 220;
    int npixel = ncols;

    tomocam::dim3_t dims = {nslices, nrows, ncols};
    tomocam::DArray<float> x1(dims);
    x1.init(1.f);
    tomocam::DArray<float> yT(dims);
    yT.init(0.f);
    auto x2 = x1;

    std::vector<float> angs(nprojs);
    for (int i = 0; i < nprojs; i++) {
        angs[i] = i * M_PI / nprojs;
    }

    // gradient 1
    tomocam::Timer t1;
    t1.start();
    auto tmp = tomocam::project(x1, angs);
    auto g1 = tomocam::backproject(tmp, angs, false);
    auto dt1 = t1.elapsed();

    // gradient 2
    // create NUFFT grids
    std::vector<tomocam::NUFFT::Grid<float>> nugrids(4);
    std::vector<tomocam::PointSpreadFunction<float>> psfs(4);
    for (int i = 0; i < 4; i++) {
        tomocam::NUFFT::Grid<float> grid(nprojs, npixel, angs.data(), i);
        nugrids[i] = grid;
        psfs[i] = tomocam::PointSpreadFunction<float>(grid);
        psfs[i].create_plans(4);
    }

    // compute gradient
    tomocam::Timer t2;
    t2.start();
    auto g2 = tomocam::gradient(x2, yT, nugrids);
    auto dt2 = t2.elapsed();


    tomocam::Timer t3;
    t3.start();
    auto g3 = tomocam::gradient2(x1, yT, psfs);
    auto dt3 = t3.elapsed();

    // report time
    std::cout << std::format("Gradient computation times (ms): g1: {}, g2: {}, g3: {}\n", dt1, dt2, dt3);

    // write to HDF5
    tomocam::h5::Writer h5fw("gradient.h5");
    h5fw.write("g1", g1);
    h5fw.write("g2", g2);

    // compare
    auto e = g1 - g2;
    auto e2 = g1 - g3;
    std::cout << "g1: " << g1.norm() << std::endl;
    std::cout << "g2: " << g2.norm() << std::endl;
    std::cout << "g3: " << g3.norm() << std::endl;
    std::cout << "Error1: " << e.norm() / g1.norm() << std::endl;
    std::cout << "Error2: " << e2.norm() / g1.norm() << std::endl;
    return 0;
}
