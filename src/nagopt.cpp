/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <cmath>
#include <cstdio>
#include <functional>

#include "dist_array.h"
#include "dist_array_ops.h"
#include "machine.h"
#include "optimize.h"
#include "tomocam.h"

#ifdef MULTIPROC
#include "multiproc.h"
#endif

namespace tomocam {

    template <typename T>
    DArray<T> nagopt(std::function<DArray<T>(DArray<T> &)> gradient,
                     std::function<T(DArray<T> &)> loss, const DArray<T> &x0,
                     T step_size, const ReconParams &params) {

        DArray<T> sol = x0;
        DArray<T> x = sol;
        DArray<T> y = sol;
        T t = 1;
        T tnew = 1;
        T step0 = step_size;
        T xerr = static_cast<T>(sol.size());

        for (size_t iter = 0; iter < params.max_iters; ++iter) {

            // update theta
            T beta = tnew * (1 / t - 1);
            tnew = static_cast<T>(0.5) *
                   (std::sqrt(std::pow(t, 4) + 4 * std::pow(t, 2)) - std::pow(t, 2));

            // update y = sol + beta * (sol - x), on the GPU
            // y, g and fy don't depend on step_size, so they're
            // invariant across the backtracking retries below
            DArray<T> d = array::axpy(x, static_cast<T>(-1), sol); // d = sol - x
            y = array::axpy(d, beta, sol); // y = beta * d + sol
            auto g = gradient(y);
            T fy = loss(y);
            T gnorm2 = array::dot(g, g);

            while (true) {

                // update x, sol = y - step_size * g
                sol = array::axpy(g, -step_size, y);

                // check if step size is small enough
                T fx = loss(sol);
                T gy = static_cast<T>(0.5) * step_size * gnorm2;

                if (fx > (fy + gy)) {
                    step_size *= static_cast<T>(0.9);
                } else {

                    // reset step size
                    step_size = step0;
                    t = tnew;

                    // compute norm of the change
                    DArray<T> dx = array::axpy(x, static_cast<T>(-1), sol);
                    xerr = array::dot(dx, dx);
#ifdef MULTIPROC
                    xerr = multiproc::mp.SumReduce(xerr);
#endif

                    x = sol;
                    break;
                }
            }
            T e = loss(sol);
#ifdef MULTIPROC
            if (multiproc::mp.first())
#endif
                // ensure that output prints in nice columns
                fprintf(stdout, "iter: %4zu, error: %5.4e, x-err: %5.4e\n", iter, e,
                        std::sqrt(xerr));
        }
        return sol;
    }

    // explicit instantiation
    template DArray<float> nagopt(std::function<DArray<float>(DArray<float> &)>,
                                  std::function<float(DArray<float> &)>,
                                  const DArray<float> &, float, const ReconParams &);
    template DArray<double> nagopt(std::function<DArray<double>(DArray<double> &)>,
                                   std::function<double(DArray<double> &)>,
                                   const DArray<double> &, double, const ReconParams &);

} // namespace tomocam
