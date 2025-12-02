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

#include <functional>
#include <iostream>
#include <optional>
#include <utility>

#include "core/internals.h"
#include "core/tomocam.h"
#include "memory/array_ops.h"
#include "memory/dist_array.h"
#include "reconstruction/optimize.h"
#include "transforms/nufft.h"
#include "utils/machine.h"

#ifdef MULTIPROC
#include "concurrency/multiproc.h"
#endif

using tomocam::DArray;
using tomocam::dim3_t;
using tomocam::preprocessing::pad2d;
using tomocam::preprocessing::postproc;
using tomocam::preprocessing::preproc;
using tomocam::transforms::PointSpreadFunction;
using tomocam::transforms::NUFFT::Grid;
using tomocam::utils::Machine::config;

namespace tomocam::reconstruction {

    template <typename T>
    DArray<T> mbir2(const DArray<T> &x_in, const DArray<T> &sino,
                    std::vector<T> angles, T center, int num_iters, T sigma, T tol,
                    T xtol) {

        // normalize
        auto maxv = array::max(sino);
#ifdef MULTIPROC
        maxv = multiproc::mp.MaxReduce(maxv);
#endif
        if (maxv == 0) { throw std::runtime_error("Sinogram is all zeros"); }
        auto sino2 = sino / maxv;
        DArray<T> x0 = x_in.clone();
        if (x_in.size() == 0) x0 = backproject(sino2, angles, true);

        // preprocess
        int nrays = sino.ncols();
        sino2 = preproc(sino2, center);
        int npad = (sino2.ncols() - x0.ncols());
        x0 = pad2d(x0, npad, PadType::SYMMETRIC);

        // recon dimensions
        int nslcs = sino2.nslices();
        int nproj = sino2.nrows();
        int ncols = sino2.ncols();

        // backproject sinogram
        auto sinoT = backproject(sino2, angles);

        // sinogram dot sinogram
        T sino_norm = array::norm2(sino2);

        // number of gpus available
        int ndevice = config.num_of_gpus();
        if (ndevice > nslcs) { ndevice = nslcs; }

        // calculate non-uniform grid for each device
        int current_dev = 0;
        SAFE_CALL(cudaGetDevice(&current_dev));
        std::vector<Grid<T>> grids(ndevice);
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            SAFE_CALL(cudaSetDevice(dev_id));
            grids[dev_id] = Grid<T>(nproj, ncols, angles.data(), dev_id);
        }

        // calculate point-spread function for each device
        std::vector<PointSpreadFunction<T>> psfs(ndevice);
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            SAFE_CALL(cudaSetDevice(dev_id));
            psfs[dev_id] = std::move(PointSpreadFunction(grids[dev_id]));
        }
        SAFE_CALL(cudaSetDevice(current_dev));

        // compute Lipschitz constant
        DArray<T> xtmp(dim3_t(1, ncols, ncols));
        DArray<T> ytmp(dim3_t(1, ncols, ncols));
        xtmp.init(1);
        ytmp.init(0);
        auto g = gradient(xtmp, ytmp, grids);
        T L = array::max(g);
#ifdef MULTIPROC
        L = multiproc::mp.MaxReduce(L);
#endif
        T step_size = 1 / L;
        if (step_size > 1) step_size = 1;
        T p = 1.2;

        // create fft plans
        int nbatch = config.slicesPerStream();
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            SAFE_CALL(cudaSetDevice(dev_id));
            psfs[dev_id].create_plans(nbatch);
        }

        // create callable functions for optimization
        auto calc_gradient = [&sinoT, &psfs, sigma, p](DArray<T> &x) -> DArray<T> {
            auto g = gradient2(x, sinoT, psfs);
            add_total_var2(x, g, sigma, p);
            return g;
        };

        auto calc_error = [&sinoT, &psfs, sino_norm](DArray<T> &x) -> T {
            auto e = residual2(x, sinoT, psfs, sino_norm);
#ifdef MULTIPROC
            e = multiproc::mp.SumReduce(e);
#endif
            return std::sqrt(e);
        };

        // create optimizer
        Optimizer<T, DArray, decltype(calc_gradient), decltype(calc_error)> opt(
            calc_gradient, calc_error);

        // run optimization
        auto rec = opt.run2(x0, num_iters, step_size, tol, xtol);
        return postproc(rec, nrays);
    }

    // explicit instantiation
    template DArray<float> mbir2(const DArray<float> &, const DArray<float> &,
                                 std::vector<float>, float, int, float, float,
                                 float);

    template DArray<double> mbir2(const DArray<double> &, const DArray<double> &,
                                  std::vector<double>, double, int, double, double,
                                  double);
} // namespace tomocam::reconstruction
