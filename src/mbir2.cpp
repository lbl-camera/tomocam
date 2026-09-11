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

#include "dist_array.h"
#include "internals.h"
#include "machine.h"
#include "nufft.h"
#include "optimize.h"
#include "tomocam.h"

#ifdef MULTIPROC
#include "multiproc.h"
#endif

namespace tomocam {

    template <typename T>
    DArray<T> mbir2(DArray<T> &x0, DArray<T> &sino, std::vector<T> angles, T center,
                    const ReconParams &params) {

        // normalize
        auto maxv = sino.max();
#ifdef MULTIPROC
        maxv = multiproc::mp.MaxReduce(maxv);
#endif
        sino /= maxv;

        // preprocess
        int nrays = sino.ncols();
        auto sino2 = preproc(sino);

        // preproc pads symmetrically, so the rotation center shifts by
        // the padding added on each side
        int npad = (sino2.ncols() - nrays) / 2;
        center += static_cast<T>(npad);

        // check for initial guess
        if (x0.size() == 0) {
            x0 = backproject(sino2, angles, center, true);
        } else {
            int npad2 = (sino2.ncols() - x0.ncols());
            x0 = pad2d(x0, npad2, PadType::SYMMETRIC);
        }

        // recon dimensions
        int nslcs = sino2.nslices();
        int nproj = sino2.nrows();
        int ncols = sino2.ncols();

        // backproject sinogram
        auto sinoT = backproject(sino2, angles, center, false);

        // number of gpus available
        int ndevice = Machine::config.num_of_gpus();

        // calculate non-uniform grid for each device
        int current_dev = 0;
        SAFE_CALL(cudaGetDevice(&current_dev));
        std::vector<PointSpreadFunction<T>> psfs;
        std::vector<nufft::Grid<T>> grids;
        psfs.reserve(ndevice);
        grids.reserve(ndevice);
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            DeviceGuard guard(dev_id);
            auto g = nufft::Grid<T>(nproj, ncols, angles.data(), dev_id);
            auto psf = PointSpreadFunction<T>(g);
            psfs.emplace_back(std::move(psf));
            grids.emplace_back(std::move(g));
        }

        // compute Lipschitz constant
        DArray<T> xtmp(dim3_t(1, ncols, ncols));
        DArray<T> ytmp(dim3_t(1, ncols, ncols));
        xtmp.init(1);
        ytmp.init(0);
        auto g = gradient2(xtmp, ytmp, psfs);
        gpu::add_tv_hessian(g, params.sigma);

        T L = g.max();
#ifdef MULTIPROC
        L = multiproc::mp.MaxReduce(L);
#endif
        T step_size = 1 / L;
        if (step_size > 1) step_size = 1;
        T p = 1.2;

        // create callable functions for optimization
        auto calc_gradient = [&sinoT, &psfs, &params, p](DArray<T> &x) -> DArray<T> {
            auto g = gradient2(x, sinoT, psfs);
            add_total_var2(x, g, static_cast<T>(params.sigma), p);
            return g;
        };

        auto calc_error = [&sino2, &grids](DArray<T> &x) -> T {
            auto e = function_value(x, sino2, grids);
#ifdef MULTIPROC
            e = multiproc::mp.SumReduce(e);
#endif
            return std::sqrt(e);
        };

        // run optimization
        auto rec = nagopt<T>(calc_gradient, calc_error, x0, step_size, params);
        return postproc(rec, nrays);
    }

    // explicit instantiation
    template DArray<float> mbir2(DArray<float> &, DArray<float> &,
                                 std::vector<float>, float, const ReconParams &);

    template DArray<double> mbir2(DArray<double> &, DArray<double> &,
                                  std::vector<double>, double, const ReconParams &);
} // namespace tomocam
