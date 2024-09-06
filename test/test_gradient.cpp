
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "dist_array.h"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "toeplitz.h"
#include "tomocam.h"


int main(int argc, char **argv) {

    const int nprojs = 200;
    const int npixel = 2047;
    const int center = npixel / 2;

    auto random = []() { return static_cast<float>(rand()) / RAND_MAX; };

    // create data
    tomocam::DArray<float> sino(tomocam::dim3_t{1, nprojs, npixel});
    for (int i = 0; i < sino.size(); i++) {
        sino[i] = random();
        // sino[i] = 0;
    }

    // create angles
    std::vector<float> angs(nprojs);
    for (int i = 0; i < nprojs; i++) { angs[i] = i * M_PI / nprojs; }

    // allocate solution array
    tomocam::dim3_t dims = {1, npixel, npixel};
    tomocam::DArray<float> x1(dims);
    x1.init(1.f);
    auto x2 = x1;

    // backproject sinogram
    auto y = tomocam::backproject(sino, angs, center);

    // gradient 1
    auto t1 = tomocam::project(x1, angs, center) - sino;
    auto g1 = tomocam::backproject(t1, angs, center);

    // gradient 2
    // create NUFFT grids
    std::vector<tomocam::PointSpreadFunction<float>> psfs(4);
    for (int i = 0; i < 4; i++) {
        tomocam::NUFFT::Grid<float> grid(nprojs, npixel, angs.data(), i);
        psfs[i] = tomocam::PointSpreadFunction<float>(grid);
    }

    ///**
    size_t free, total;
    SAFE_CALL(cudaSetDevice(0));
    tomocam::DeviceArray<float> dx2(x2.dims());
    cudaMemcpy(dx2.dev_ptr(), x2.begin(), x2.bytes(), cudaMemcpyHostToDevice);
    for (int i = 0; i < 100; i++) {
        psfs[0].convolve(dx2);
        SAFE_CALL(cudaMemGetInfo(&free, &total));
        std::cout << "Free: " << free / 1024 / 1024 << " MB" << std::endl;
    }
    //*/

    // compute gradient
    auto g2 = tomocam::gradient2(x2, y, psfs);

    // write to HDF5
    tomocam::h5::Writer h5fw("gradient.h5");
    h5fw.write("g1", g1);
    h5fw.write("g2", g2);

    // compare
    auto e = g1 - g2;
    std::cout << "Error: " << e.norm() / g1.norm() << std::endl;
    return 0;
}
