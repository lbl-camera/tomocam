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
    int nproj = 360;
    int ncols = 511;

    // create a hdf5 writer
    tomocam::h5::H5Writer fp("test_toeplitz.h5");

    tomocam::dim3_t dim1 = {1, nproj, ncols};
    tomocam::dim3_t dim2 = {1, ncols, ncols};

    // set seed
    srand(101);

    // generate random data
    tomocam::DArray<float> f(dim2);
    for (int i = 0; i < f.size(); i++) f[i] = random<float>();

    // create device array
    tomocam::DeviceArray<float> d_f(dim2);
    cudaMemcpy(d_f.dev_ptr(), f.begin(), f.bytes(), cudaMemcpyHostToDevice);

    // angles 0 - 180
    std::vector<float> theta(nproj);
    for (int i = 0; i < nproj; i++) theta[i] = i * M_PI / nproj;

    auto nugrid = tomocam::NUFFT::Grid(nproj, ncols, theta.data(), 0);
    auto psf = tomocam::PointSpreadFunction(nugrid);

    // compute gradient using toeplitz
    auto df = psf.convolve(d_f, cudaStreamPerThread);

    // copy data to host
    tomocam::DArray<float> g(df.dims());
    cudaMemcpy(g.begin(), df.dev_ptr(), g.bytes(), cudaMemcpyDeviceToHost);

    // write data to hdf5
    fp.write("toeplitz", g);

    // compute gradient using nufft
    auto dfc = tomocam::complex(d_f, cudaStreamPerThread);
    tomocam::DeviceArray<tomocam::gpu::complex_t<float>> d_c(dim1);
    tomocam::DeviceArray<tomocam::gpu::complex_t<float>> df2(dim2);
    tomocam::NUFFT::nufft2d2(d_c, dfc, nugrid);
    tomocam::NUFFT::nufft2d1(d_c, df2, nugrid);
    auto dg2 = tomocam::real(df2, cudaStreamPerThread);

    tomocam::DArray<float> g2(df2.dims());
    // copy data to host
    cudaMemcpy(g2.begin(), dg2.dev_ptr(), g2.bytes(), cudaMemcpyDeviceToHost);

    fp.write("nufft", g2);

    auto err = g - g2;
    std::cout << "Error: " << err.norm() / g2.norm() << std::endl;

    return 0;
}

