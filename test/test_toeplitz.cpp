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

template <typename T>
T random() {
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}

int main(int argc, char **argv) {

    // read data
    int nproj = 5;
    int ncols = 5;

    // create a hdf5 writer
    tomocam::h5::H5Writer fp("test_toeplitz.h5");

    tomocam::dim3_t dim1 = {1, nproj, ncols};
    tomocam::dim3_t dim2 = {1, ncols, ncols};

    // set seed
    srand(101);

    // generate random data
    tomocam::DArray<float> f1(dim1);
    for (int i = 0; i < f1.size(); i++) f1[i] = random<float>();

    // initialize solution
    tomocam::DArray<float> f(dim2);
    for (int i = 0; i < f.size(); i++) f[i] = 1.0; // random<float>();

    // angles 0 - 180
    std::vector<float> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * M_PI / nproj;

    // calculate backprojection of data
    auto yT = tomocam::backproject(f1, theta);

    return 0;
}

