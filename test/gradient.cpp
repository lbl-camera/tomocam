
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "core/tomocam.h"
#include "io/hdf5/writer.h"
#include "memory/array_ops.h"
#include "memory/dist_array.h"
#include "transforms/toeplitz.h"
#include "utils/timer.h"

using json = nlohmann::json;

using tomocam::DArray;
using tomocam::dim3_t;
using tomocam::io::h5::Writer;
using tomocam::transforms::PointSpreadFunction;
using tomocam::transforms::NUFFT::Grid;
using tomocam::utils::Timer;

int main(int argc, char **argv) {

    // define size of the reconstruction
    int nslices = 16;
    int nrows = 2047;
    int ncols = 2047;

    // allocate solution array
    int nprojs = 220;
    int npixel = ncols;

    dim3_t dims = {nslices, nrows, ncols};
    DArray<float> x1(dims);
    x1.init(1.f);
    DArray<float> yT(dims);
    yT.init(0.f);
    auto x2 = x1.clone();

    std::vector<float> angs(nprojs);
    for (int i = 0; i < nprojs; i++) { angs[i] = i * M_PI / nprojs; }

    // gradient 1
    Timer t1;
    t1.start();
    auto tmp = tomocam::project(x1, angs);
    auto g1 = tomocam::backproject(tmp, angs, false);
    auto dt1 = t1.elapsed();

    // gradient 2
    // create NUFFT grids
    std::vector<Grid<float>> nugrids(4);
    std::vector<PointSpreadFunction<float>> psfs(4);
    for (int i = 0; i < 4; i++) {
        Grid<float> grid(nprojs, npixel, angs.data(), i);
        nugrids[i] = grid;
        psfs[i] = PointSpreadFunction<float>(grid);
        psfs[i].create_plans(4);
    }

    // compute gradient
    Timer t2;
    t2.start();
    auto g2 = tomocam::gradient(x2, yT, nugrids);
    auto dt2 = t2.elapsed();

    Timer t3;
    t3.start();
    auto g3 = tomocam::gradient2(x1, yT, psfs);
    auto dt3 = t3.elapsed();

    // report time
    std::cout << std::format("g1: {} ms, g2: {} ms, g3: {} ms\n", dt1, dt2, dt3);

    // write to HDF5
    Writer h5fw("gradient.h5");
    h5fw.write("g1", g1);
    h5fw.write("g2", g2);

    // compare
    auto e = g1 - g2;
    auto e2 = g1 - g3;
    std::cout << std::format("|g1|_2:  {:.6e}\n", tomocam::array::norm2(g1));
    std::cout << std::format("|g2|_2:  {:.6e}\n", tomocam::array::norm2(g2));
    std::cout << std::format("|g3|_2:  {:.6e}\n", tomocam::array::norm2(g3));
    std::cout << std::format("|e|_2:  {:.6e}\n", tomocam::array::norm2(e));
    std::cout << std::format("|e2|_2:  {:.6e}\n", tomocam::array::norm2(e2));
    return 0;
}
