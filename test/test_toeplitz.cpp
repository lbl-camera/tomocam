#include <iostream>
#include <fstream>
#include <ctime>
#include <array>
#include <random>

#include <cuda_runtime.h>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "toeplitz.h"
#include "tomocam.h"

#define USE_DOUBLE1

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

template <typename T>
T random() {
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}

int main(int argc, char **argv) {

    // read data
    int nproj = 360;
    int ncols = 511;
    real_t center = static_cast<real_t>(ncols) / 2;

    // create a hdf5 writer
    tomocam::h5::Writer fp("test_toeplitz.h5");
    tomocam::dim3_t dim1 = {1, nproj, ncols};
    tomocam::dim3_t dim2 = {1, ncols, ncols};

    // generate random data
    tomocam::DArray<real_t> y(dim1);
    for (int i = 0; i < y.size(); i++) y[i] = random<real_t>();

    // initialize solution
    tomocam::DArray<real_t> f(dim2);
    for (int i = 0; i < f.size(); i++) f[i] = 1.0;

    // angles 0 - 180 degrees
    std::vector<real_t> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * M_PI / nproj;

    // create a nugrid and psfs
    int ndevices = 4;
    std::vector<tomocam::NUFFT::Grid<real_t>> nugrids(ndevices);
    for (int i = 0; i < ndevices; i++)
        nugrids[i] =
            tomocam::NUFFT::Grid<real_t>(nproj, ncols, theta.data(), i);

    std::vector<tomocam::PointSpreadFunction<real_t>> psfs(ndevices);
    for (int i = 0; i < ndevices; i++)
        psfs[i] = tomocam::PointSpreadFunction<real_t>(nugrids[i]);

    // calculate backprojection of data
    auto yT = tomocam::backproject(y, theta, center);

    // calculate classical gradient
    auto g1 = tomocam::gradient(f, y, nugrids, center);

    // calculate gradient using toeplitz matrix
    auto g2 = tomocam::gradient2(f, yT, psfs);

    // compare the two gradients
    auto err = std::sqrt((g1 - g2).norm()) / std::sqrt(g1.norm());
    std::cout << "Relative error: " << err << std::endl;
    return 0;
}

