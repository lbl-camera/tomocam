/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 *Software to reproduce, distribute copies to the public, prepare derivative
 *works, and perform publicly and display publicly, and to permit other to do
 *so.
 *---------------------------------------------------------------------------------
 */

#include <iostream>
#include <utility>
#include <functional>
#include <pybind11/pybind11.h>

#include "dist_array.h"
#include "optimize.h"
#include "tomocam.h"
#include "machine.h"
#include "nufft.h"

namespace tomocam {

    template <typename T>
    DArray<T> mbir2(DArray<T> &sino, std::vector<T> angles, T center, T sigma,
        T p, int num_iters, T step_size, T tol, T penalty) {

        // pad and shift sinogram
        int nrays = sino.ncols();
        sino = preproc(sino, center);

        // recon dimensions
        int nslcs = sino.nslices();
        int nproj = sino.nrows();
        int ncols = sino.ncols();

        dim3_t dims(nslcs, ncols, ncols);
        DArray<T> sinoT = backproject(sino, angles, center);

        // sinogram dot sinogram
        T sino_norm = sino.norm();

        // initialize x0
        DArray<T> x0(dims);
        x0.init(1);

        // number of gpus available
        int ndevice = Machine::config.num_of_gpus();
        if (ndevice > nslcs) { ndevice = 1; }

        // calculate point-spread function for each device
        std::vector<PointSpreadFunction<T>> psfs(ndevice);
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            SAFE_CALL(cudaSetDevice(dev_id));
            auto nugrid = NUFFT::Grid(nproj, ncols, angles.data(), dev_id);
            psfs[dev_id] = PointSpreadFunction(nugrid);
        }

        // compute Lipschitz constant
        DArray<T> xtmp(dim3_t(1, ncols, ncols));
        DArray<T> ytmp(dim3_t(1, ncols, ncols));
        xtmp.init(1);
        ytmp.init(0);
        auto g = gradient2(xtmp, ytmp, psfs);
        gpu::add_tv_hessian(g, sigma);
        T L = g.max();
        step_size = step_size / L;

        // create callable functions for optimization
        auto calc_gradient = [&sinoT, &psfs, p, sigma](DArray<T> &x) -> DArray<T> {
            auto g = gradient2(x, sinoT, psfs);
            add_total_var(x, g, p, sigma);
            return g;
        };

        auto calc_error = [&sinoT, &psfs, sino_norm](DArray<T> &x) -> T {
            return function_value2(x, sinoT, psfs, sino_norm);
        };

        // create optimizer
        Optimizer<T, DArray, decltype(calc_gradient), decltype(calc_error)> opt(
            calc_gradient, calc_error);

        // run optimization
        auto rec = opt.run2(x0, num_iters, step_size, tol);
        return postproc(rec, nrays);
    }

    // explicit instantiation
    template DArray<float> mbir2(DArray<float> &, std::vector<float>, float,
        float, float, int, float, float, float);
    template DArray<double> mbir2(DArray<double> &, std::vector<double>, double,
        double, double, int, double, double, double);
} // namespace tomocam
