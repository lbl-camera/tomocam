#include <array>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "timer.h"
#include "toeplitz.h"
#include <cuda_runtime.h>

#include "timer.h"
#include "tomocam.h"

// #define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

int main(int argc, char **argv) {

    // read data
    int nproj = 180;
    int ncols = 511;
    if (argc > 1) {
        int nc = std::atoi(argv[1]);
        ncols = std::max(511, nc);
        nproj = std::max(180, ncols / 10);
    }
    real_t center = static_cast<real_t>(ncols) / 2;

    // curr gpu device
    int cuda_device = 0;
    cudaGetDevice(&cuda_device);

    // dimensions of data and solution
    tomocam::dim3_t dim1 = {16, nproj, ncols};
    tomocam::dim3_t dim2 = {16, ncols, ncols};

    // generate random data
    auto rng = NPRandom();
    tomocam::DArray<real_t> y(dim1);
    for (int i = 0; i < y.size(); i++) y[i] = rng.rand<real_t>();

    // initialize solution
    tomocam::DArray<real_t> f(dim2);
    for (int i = 0; i < f.size(); i++) f[i] = 1.0;

    // angles 0 - 180 degrees
    std::vector<real_t> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * M_PI / nproj;

    // create a nugrid and psfs
    int ndevices = 4;
    std::vector<tomocam::NUFFT::Grid<real_t>> nugrids(ndevices);
    for (int i = 0; i < ndevices; i++) {
        cudaSetDevice(i);
        nugrids[i] = tomocam::NUFFT::Grid<real_t>(nproj, ncols, theta.data(), i);
    }
    cudaSetDevice(cuda_device);

    std::vector<tomocam::PointSpreadFunction<real_t>> psfs(ndevices);
    for (int i = 0; i < ndevices; i++) {
        cudaSetDevice(i);
        psfs[i] = tomocam::PointSpreadFunction<real_t>(nugrids[i]);
    }
    cudaSetDevice(cuda_device);

    // calculate backprojection of data
    auto yT = tomocam::backproject(y, theta, false);

    // calculate gradient using direct method
    Timer t1;
    t1.start();
    auto e1 = tomocam::project(f, theta) - y;
    auto g1 = tomocam::backproject(e1, theta, false);
    t1.stop();

    // calculate gradient using toeplitz matrix
    Timer t2;
    t2.start();
    auto g2 = tomocam::gradient2(f, yT, psfs);
    t2.stop();

    std::cout << "g1.norm(): " << g1.norm() << std::endl;
    std::cout << "g2.norm(): " << g2.norm() << std::endl;

    fprintf(stdout, "Time taken(ms): direct method: %d\n", t1.ms());
    fprintf(stdout, "Time taken(ms): toeplitz method: %d\n", t2.ms());

    // compare the two gradients
    auto err = std::sqrt((g1 - g2).norm()) / std::sqrt(g1.norm());
    std::cout << "Relative error: " << err << std::endl;
    return 0;
}
