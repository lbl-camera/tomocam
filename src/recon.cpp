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
#include <optional>
#include <tuple>
#include <vector>

#include "dist_array.h"
#include "padding.h"
#include "tomocam.h"

#ifdef MULTIPROC
#include "multiproc.h"
#endif

namespace tomocam {

    template <typename T>
    DArray<T> recon(const DArray<T> &sino, const std::vector<T> &theta,
        T center, int num_iters, T sigma, T tol, T xtol, bool hierarchical,
        DArray<T> x0) {

        // downsampling factor
        int skip = 4;

        // normalize
        auto maxv = sino.max();
#ifdef MULTIPROC
        maxv = multiproc::mp.MaxReduce(maxv);
#endif
        if (maxv == 0) {
            throw std::runtime_error(
                "Error: Maximum value of sinogram is zero.");
        }

        auto sino2 = sino / maxv;

        if (hierarchical) {

            int iters = num_iters;
            for (int skip = 4; skip >= 0; skip -= 2) {

                // downsample the sinogram
                auto sino3 = downsample(sino2, skip);

                // check if the user has not specified the size of the
                // reconstruction array
                if (x0.size() == 0) {
                    x0 = DArray<T>(
                        dim3_t(sino3.nslices(), sino3.ncols(), sino3.ncols()));
                    x0.init(1);
                }

                // if x0 has a non-zero size, pad it to the size of the sinogram
                int npad = (sino3.ncols() - x0.ncols());
                if (npad > 0) { x0 = pad2d(x0, npad, PadType::SYMMETRIC); }

                T cen = center * static_cast<T>(sino3.ncols()) /
                        static_cast<T>(sino.ncols());
                if (skip > 0) cen = std::floor(cen + 0.5);

                x0 = mbir(x0, sino3, theta, cen, iters, sigma, tol, xtol);
                iters = iters / 2;

                // upsample the reconstruction by a factor of 2
                int nslcs = 2 * sino3.nslices();
                int ncols = 2 * sino3.ncols() - 1;
                if (skip > 0) x0 = upsample(x0, dim3_t(nslcs, ncols, ncols));
            }
        } else {

            // check if the user has not specified the size of the
            // reconstruction array
            if (x0.size() == 0) {
                x0 = DArray<T>(
                    dim3_t(sino2.nslices(), sino2.ncols(), sino2.ncols()));
                x0.init(1);
            }

            // if user has passed in an initial guess, pad it to the size of the
            // sinogram
            int npad = (sino2.ncols() - x0.ncols());
            if (npad > 0) { x0 = pad2d(x0, npad, PadType::SYMMETRIC); }

            x0 = mbir2(x0, sino2, theta, center, num_iters, sigma, tol, xtol);
        }
        return x0;
    }

    // wrapper function without x0
    template <typename T>
    DArray<T> recon(const DArray<T> &sino, const std::vector<T> &theta,
        T center, int num_iters, T sigma, T tol, T xtol, bool hierarchical) {
        // create a default reconstruction array
        DArray<T> x0({0, 0, 0});
        return recon(sino, theta, center, num_iters, sigma, tol, xtol,
            hierarchical, x0);
    }
    // explicit instantiation
    template DArray<float> recon(const DArray<float> &,
        const std::vector<float> &, float, int, float, float, float, bool);
    template DArray<double> recon(const DArray<double> &,
        const std::vector<double> &, double, int, double, double, double, bool);

    // wrapper function without herarchical flag, but with x0
    template <typename T>
    DArray<T> recon(const DArray<T> &sino, const std::vector<T> &theta,
        T center, int num_iters, T sigma, T tol, T xtol, DArray<T> x0) {
        return recon(sino, theta, center, num_iters, sigma, tol, xtol, false,
            x0);
    }
    // explicit instantiation
    template DArray<float> recon(const DArray<float> &,
        const std::vector<float> &, float, int, float, float, float,
        DArray<float>);
    template DArray<double> recon(const DArray<double> &,
        const std::vector<double> &, double, int, double, double, double,
        DArray<double>);

    // wrapper function without herarchical flag and without x0
    template <typename T>
    DArray<T> recon(const DArray<T> &sino, const std::vector<T> &theta,
        T center, int num_iters, T sigma, T tol, T xtol) {
        DArray<T> x0({0, 0, 0});
        return recon(sino, theta, center, num_iters, sigma, tol, xtol, false,
            x0);
    }
    // explicit instantiation
    template DArray<float> recon(const DArray<float> &,
        const std::vector<float> &, float, int, float, float, float);
    template DArray<double> recon(const DArray<double> &,
        const std::vector<double> &, double, int, double, double, double);

} // namespace tomocam
