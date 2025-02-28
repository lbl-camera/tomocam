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
#include "machine.h"
#include "nufft.h"
#include "optimize.h"
#include "tomocam.h"
#include "utils.h"

#ifdef MULTIPROC
#include "multiproc.h"
#endif

namespace tomocam {

    template <typename T>
    DArray<T> mbir(DArray<T> &x0, DArray<T> &sino, std::vector<T> angles, T center,
        int num_iters, T sigma, T tol, T xtol) {

        // normalize
        auto maxv = sino.max();
        #ifdef MULTIPROC
        maxv = multiproc::mp.MaxReduce(maxv);
        #endif
        sino /= maxv;

        // pad and center
        int nrays = sino.ncols();
        sino = preproc(sino, center);
        int npad = (sino.ncols() - x0.ncols());
        x0 = pad2d(x0, npad, PadType::SYMMETRIC);

        // backproject sinogram
        auto sinoT = backproject(sino, angles, center);

        // recon dimensions
        int nslcs = sino.nslices();
        int nproj = sino.nrows();
        int ncols = sino.ncols();

        // number of gpus available
        int ndevice = Machine::config.num_of_gpus();
        if (ndevice > nslcs) {
            ndevice = 1;
        }

        // calculate point-spread function for each device
        int current_dev = 0;
        SAFE_CALL(cudaGetDevice(&current_dev));
        std::vector<NUFFT::Grid<T>> grids(ndevice);
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            grids[dev_id] = NUFFT::Grid<T>(nproj, ncols, angles.data(), dev_id);
        }
        SAFE_CALL(cudaSetDevice(current_dev));

        // compute Lipschitz constant
        DArray<T> xtmp(dim3_t(1, ncols, ncols));
        DArray<T> ytmp(dim3_t(1, ncols, ncols));
        xtmp.init(1);
        ytmp.init(0);
        auto g = gradient(xtmp, ytmp, grids, center);
        gpu::add_tv_hessian(g, sigma);
        T L = g.max();

        #ifdef MULTIPROC
        L = multiproc::mp.MaxReduce(L);
        #endif
        T step_size = 1 / L;
        T p = 1.2;

        // create callable functions for optimization
        auto calc_gradient = [&sinoT, &grids, center, sigma, p](DArray<T> &x) -> DArray<T> {
            auto g = gradient(x, sinoT, grids, center);
            add_total_var(x, g, sigma, p);
            return g;
        };

        auto calc_error = [&sino, &grids, center](DArray<T> &x) -> T {
            T e = function_value(x, sino, grids, center);
            double size = static_cast<double>(x.size());
            #ifdef MULTIPROC
            e = multiproc::mp.SumReduce(e);
            size = multiproc::mp.SumReduce(size);
            #endif
            return e;
        };

        // create optimizer
        Optimizer<T, DArray, decltype(calc_gradient), decltype(calc_error)>
        opt(calc_gradient, calc_error);

        // run optimization
        auto recon = opt.run2(x0, num_iters, step_size, tol, xtol);
        return postproc(recon, nrays);
    }

// explicit instantiation
    template DArray<float> mbir(DArray<float> &, DArray<float> &, std::vector<float>, float,
        int, float, float, float);
    template DArray<double> mbir(DArray<double> &, DArray<double> &, std::vector<double>,
        double, int, double, double, double);
} // namespace tomocam
