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

    // CT reconstruction via split-Bregman TV regularization. Shares the
    // normalize/preproc/backproject/PSF setup with mbir2, but solves
    //   argmin_x ||Ax - b||^2 + lambda * TV(x)
    // by handing split_bregman the *normal* operator (A^TA, via the
    // Toeplitz PSF trick already used by gradient2) and the backprojected
    // sinogram (A^T b), rather than FISTA/nagopt.
    template <typename T>
    DArray<T> mbir_bregman(DArray<T> &x0, DArray<T> &sino,
                           std::vector<T> angles, T center, int num_iters,
                           T tol, T xtol) {

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
        int nproj = sino2.nrows();
        int ncols = sino2.ncols();

        // backproject sinogram -> A^T b, in image space
        auto sinoT = backproject(sino2, angles, center, false);

        // zero image-space array, so gradient2(x, zeros, psfs) reduces to
        // the pure normal operator A^TA x (no -A^T b term)
        DArray<T> zeros(sinoT.dims());
        zeros.init(T(0));

        // number of gpus available
        int ndevice = Machine::config.num_of_gpus();

        // calculate non-uniform grid for each device
        int current_dev = 0;
        SAFE_CALL(cudaGetDevice(&current_dev));
        std::vector<PointSpreadFunction<T>> psfs;
        psfs.reserve(ndevice);
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            SAFE_CALL(cudaSetDevice(dev_id));
            auto g = nufft::Grid<T>(nproj, ncols, angles.data(), dev_id);
            psfs.emplace_back(PointSpreadFunction<T>(g));
        }

        // A^TA, via the Toeplitz PSF trick (same as mbir2's gradient2)
        std::function<DArray<T>(DArray<T> &)> Anorm =
            [&psfs, &zeros](DArray<T> &x) -> DArray<T> {
            return gradient2(x, zeros, psfs);
        };

        // split-Bregman / CG parameters. CG is capped at a single inner
        // iteration per outer step (linearized-Bregman style), so `tol`
        // never gets a chance to trigger early exit -- kept for parity
        // with mbir2's CLI parameters.
        Params params;
        params.max_iters = 1;
        params.tol = tol;
        params.xtol = xtol;
        params.outer_max = static_cast<size_t>(num_iters);

        auto rec = split_bregman<T>(Anorm, sinoT, x0, params);
        return postproc(rec, nrays);
    }

    // explicit instantiation
    template DArray<float> mbir_bregman(DArray<float> &, DArray<float> &,
                                        std::vector<float>, float, int, float,
                                        float);
    template DArray<double> mbir_bregman(DArray<double> &, DArray<double> &,
                                         std::vector<double>, double, int,
                                         double, double);
} // namespace tomocam
