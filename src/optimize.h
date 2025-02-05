/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <cmath>
#include <limits>
#include <tuple>
#include <format>

#include "dist_array.h"
#include "tomocam.h"
#include "machine.h"

#include "debug.h"

#ifndef TOMOCAM_OPTIMIZE__H
#define TOMOCAM_OPTIMIZE__H

namespace tomocam {

    template <typename T, template <typename> class Array, typename Gradient,
        typename Error>
    class Optimizer {
        private:
            Gradient gradient_;
            Error error_;

        public:
            // constructor
            Optimizer(Gradient gradient, Error error) :
                gradient_(gradient), error_(error) {}

            // fixed step-size
            Array<T> run(Array<T> sol, int max_iters, T step_size, T tol) {

                // initialize
                Array<T> x = sol;
                Array<T> y = sol;
                T t = 1;
                T tnew = 1;
                T xerr = static_cast<T>(sol.size());

                // set error to infinity
                T e_old = std::numeric_limits<T>::infinity();
                for (int iter = 0; iter < max_iters; iter++) {

                    T beta = tnew * (1 / t - 1);
                    y = sol + (sol - x) * beta;
                    auto g = gradient_(y);

                    x = sol;
                    sol = y - g * step_size;

                    // update theta
                    T temp = 0.5 * (std::sqrt(std::pow(t, 4)
                        + 4 * std::pow(t, 2))
                        - std::pow(t, 2));
                    t = tnew;
                    tnew = temp;

                    auto e = error_(sol);
                    if (e > e_old) {
                        g = gradient_(x);
                        sol = x - g * step_size;
                        xerr = (sol - x).norm();
                        e = error_(sol);
                    }
                    e_old = e;
                    #ifdef MULTIPROC
                    if (multiproc::mp.first())
                    #endif
                        // ensure that output prints in nice columns
                        std::cout << std::format("iter: {:10}, error: {:10}, x-err: {:10}",
                            iter, e, std::sqrt(xerr)) << std::endl;
                }
                return sol;
            }

            Array<T> run2(Array<T> sol, int max_iters, T step_size, T tol, T xtol) {

                // initialize
                Array<T> x = sol;
                Array<T> y = sol;
                T t = 1;
                T tnew = 1;
                T step0 = step_size;
                T xerr = static_cast<T>(sol.size());

                for (int iter = 0; iter < max_iters; iter++) {
                    while (true) {

                        // update theta
                        T beta = tnew * (1 / t - 1);
                        tnew = 0.5 * (std::sqrt(std::pow(t, 4)
                            + 4 * std::pow(t, 2))
                            - std::pow(t, 2));

                        // update y
                        y = sol + (sol - x) * beta;
                        auto g = gradient_(y);

                        // update x
                        sol = y - g * step_size;

                        // check if step size is small enough
                        T fx = error_(sol);
                        T fy = error_(y);
                        T gy = 0.5 * step_size * g.norm();
                        if (fx > (fy + gy))
                            step_size *= 0.9;
                        else {
                            step_size = step0;
                            t = tnew;
                            xerr = (sol - x).norm();
                            #ifdef MULTIPROC
                            xerr = multiproc::mp.SumReduce(xerr);
                            #endif
                            x = sol;
                            break;
                        }
                    }
                    T e = error_(sol);
                    #ifdef MULTIPROC
                    if (multiproc::mp.first())
                    #endif
                        // ensure that output prints in nice columns
                        std::cout << std::format("iter: {:4}, error: {:5.4e}, x-err: {:5.4e}",
                            iter, e, std::sqrt(xerr)) << std::endl;
                }
                return sol;
            }
    };

} // namespace tomocam

#endif // TOMOCAM_OPTIMIZE__H
