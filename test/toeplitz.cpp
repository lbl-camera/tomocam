#include <array>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>

#include "core/tomocam.h"
#include "io/hdf5/writer.h"
#include "memory/array_ops.h"
#include "memory/dist_array.h"
#include "transforms/toeplitz.h"
#include "utils/random.h"
#include "utils/timer.h"

#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

using tomocam::transforms::PointSpreadFunction;
using tomocam::transforms::NUFFT::Grid;
using tomocam::utils::NPRandom;
using tomocam::utils::Timer;

int main(int argc, char **argv) {

    // read data
    int nproj = 360;
    int ncols = 2047;
    real_t center = static_cast<real_t>(ncols) / 2;

    // create a hdf5 writer
    tomocam::io::h5::Writer fp("test_toeplitz.h5");
    tomocam::dim3_t dim1 = {1, nproj, ncols};
    tomocam::dim3_t dim2 = {1, ncols, ncols};

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
    std::vector<Grid<real_t>> nugrids(ndevices);
    for (int i = 0; i < ndevices; i++)
        nugrids[i] = Grid<real_t>(nproj, ncols, theta.data(), i);

    std::vector<PointSpreadFunction<real_t>> psfs(ndevices);
    for (int i = 0; i < ndevices; i++)
        psfs[i] = PointSpreadFunction<real_t>(nugrids[i]);

    // calculate backprojection of data
    auto yT = tomocam::backproject(y, theta, false);

    Timer t1;
    t1.start();

    // calculate classical gradient
    auto g1 = tomocam::gradient(f, yT, nugrids);

    t1.stop();

    // calculate gradient using toeplitz matrix
    Timer t2;
    t2.start();
    auto g2 = tomocam::gradient2(f, yT, psfs);
    t2.stop();

    std::cout << std::format("|g1|_2: {}\n", tomocam::array::norm2(g1));
    std::cout << std::format("|g2|_2: {}\n", tomocam::array::norm2(g2));

    std::cout << std::format("Time taken(ms): regular method: {}\n", t1.ms());
    std::cout << std::format("Time taken(ms): toeplitz method: {}\n", t2.ms());

    // compare the two gradients
    auto diff_norm = std::sqrt(tomocam::array::norm2(g1 - g2));
    auto norm_g1 = std::sqrt(tomocam::array::norm2(g1));
    std::cout << std::format("Relative error: {}\n", diff_norm / norm_g1);
    return 0;
}
