
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "machine.h"
#include "toeplitz.h"
#include "timer.h"
#include "tomocam.h"

using json = nlohmann::json;
int main(int argc, char **argv) {

    // define size of the reconstruction
    int nslices = 16;
    int nrows = 2047;
    int ncols = 2047;
    float center = (float)ncols / 2;

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
    for (int i = 0; i < nprojs; i++) { angs[i] = i * M_PI / nprojs; }

    // gradient 1
    tomocam::Timer t1;
    t1.start();
    auto tmp = tomocam::project(x1, angs);
    auto g1 = tomocam::backproject(tmp, angs, center, false);
    t1.stop();
    auto dt1 = t1.ms();

    // gradient 2
    // create nufft grids

    std::vector<tomocam::nufft::Grid<float>> nugrids;
    std::vector<tomocam::PointSpreadFunction<float>> psfs;
    int ndevices = tomocam::Machine::config.num_of_gpus();
    int current_device = 0;
    cudaGetDevice(&current_device);
    for (int i = 0; i < ndevices; i++) {
        cudaSetDevice(i);
        auto grid = tomocam::nufft::Grid<float>(nprojs, npixel, angs.data(), i);
        auto psf = tomocam::PointSpreadFunction<float>(grid);
        psfs.emplace_back(std::move(psf));
        nugrids.emplace_back(std::move(grid));
    }
    cudaSetDevice(current_device);

    // compute gradient
    tomocam::Timer t2;
    t2.start();
    auto g2 = tomocam::gradient(x2, yT, nugrids);
    t2.stop();
    auto dt2 = t2.ms();

    tomocam::Timer t3;
    t3.start();
    auto g3 = tomocam::gradient2(x1, yT, psfs);
    t3.stop();
    auto dt3 = t3.ms();

    // report time
    std::cout << std::format("Gradient computation times (ms): direct-method: {}, "
                             "nufft_cached: {}, toeplitz: {}\n",
                             dt1, dt2, dt3);
    // write to HDF5
    tomocam::h5::Writer h5fw("gradient.h5");
    h5fw.write("direct_method", g1);
    h5fw.write("nufft_cached", g2);

    // compare
    auto e = g1 - g2;
    auto e2 = g1 - g3;
    std::cout << "direct_method: " << g1.norm() << std::endl;
    std::cout << "nufft_cached: " << g2.norm() << std::endl;
    std::cout << "toeplitz: " << g3.norm() << std::endl;
    std::cout << "Error1 (direct-nufft_cached).norm() / direct.norm(): "
              << e.norm() / g1.norm() << std::endl;
    std::cout << "Error2:(direct-toeplitz).norm() / direct.norm() "
              << e2.norm() / g1.norm() << std::endl;

    return 0;
}
