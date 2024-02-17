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

#include "dist_array.h"
#include "tomocam.h"
#include "machine.h"

const float inf = std::numeric_limits<float>::infinity();

#ifndef TOMOCAM_OPTIMIZE__H
#define TOMOCAM_OPTIMIZE__H

namespace tomocam {

    template <typename T>
    inline DArray<T> calc_gradient(DArray<T> &x, DArray<T> &sino,
            float * angles, float center, float oversample, 
            float p, float sigma, float lam) {
        DArray<T> g = x;
        gradient(g, sino, angles, center, oversample);
        add_total_var(x, g, p, sigma, lam);
        return g;
    }

    template <typename T>
    inline T calc_lipschitz(DArray<T> &x, DArray<T> &sino,
            float * angles, float center, float oversample, 
            float p, float sigma, float lam) {
        DArray<T> g(x.dims());
        DArray<T> y(sino.dims());
        radon(x, y, angles, center, oversample);
        iradon(y, g, angles, center, oversample);
        add_total_var(x, g, p, sigma, lam);
        return g.max();
    }

    template <typename T>
    inline T error(DArray<T> &x, DArray<T> &sino, 
            float * angles, float center, float oversample,
            float p, float sigma, float lam) {
        DArray<T> g(sino.dims());
        radon(x, g, angles, center, oversample);
        return (g - sino).norm();
    }

    template<typename T>
    class Optimizer {
      private:
        int max_iters_;
        T tol_;
        dim3_t dims_;

      public:

        Optimizer(dim3_t d, int niters, T tol = inf):
            dims_(d), max_iters_(niters), tol_(tol) {}

        DArray<T> minimize(DArray<T> &sino, float *angles, float center,
                float oversample, float p, float sigma, float lam) {

            // initialize 
            DArray<T> x(dims_);
            x.init(1.);
            DArray<T> xold(dims_);
            xold.init(1.);
            DArray<T> y(dims_);

            T t = 1;
            T tnew = 1; 
            // compute Lipschitz
            T lipschitz_ = calc_lipschitz(xold, sino, angles, center, oversample, p, sigma, lam);
            // gradient step size
            T step = 0.1/lipschitz_;
            T e_old = inf;
            for (int iter = 0; iter < max_iters_; iter++) {
                T beta = tnew * (1/t - 1);
                y = x + (x - xold) * beta;
                auto g = calc_gradient(y, sino, angles, center, oversample, p, sigma, lam);
                xold = x;
                x = y - g * step; 
                // update theta
                T temp = 0.5 * (std::sqrt(std::pow(t,4) 
                            + 4 * std::pow(t,2))
                        - std::pow(t,2));
                t = tnew;
                tnew = temp;
               
                T e = error(x, sino, angles, center, oversample, p, sigma, lam);
                if (e > e_old) {
                    g = calc_gradient(xold, sino, angles, center, oversample, p, sigma, lam);
                    x = xold - g * step;
                    e = error(x, sino, angles, center, oversample, p, sigma, lam);
                }
                e_old = e;
                std::cout << "iter: " << iter << ", error: " << e << std::endl;
            }
            return x;
        }

        DArray<T> minimize2(DArray<T> &sino, float *angles, float center,
                float oversample, float p, float sigma, float lam) {

            // initialize 
            DArray<T> x(dims_);
            x.init(1.);
            DArray<T> xold(dims_);
            xold.init(1.);
            DArray<T> y(dims_);

            T t = 1;
            T tnew = 1; 
            // compute Lipschitz
            T lipschitz_ = calc_lipschitz(xold, sino, angles, center, oversample, p, sigma, lam);

            // gradient step size
            T step = 0.1/lipschitz_;
            T step_prev = step;

            for (int iter = 0; iter < max_iters_; iter++) {
                while (true) {

                    // update theta
                    T beta = tnew * (1/t - 1);
                    T a2 = t * t * step / step_prev;
                    tnew = 0.5 * (std::sqrt(std::pow(t,4) 
                             + 4 * std::pow(t,2))
                         - std::pow(t,2));

                    // update y
                    y = x + (x - xold) * beta;
                    auto g = calc_gradient(y, sino, angles, center, oversample, p, sigma, lam);
                   
                    // update x
                    x = y - g * step; 
                 
                    // check if step size is small enough
                    T fx = error(x, sino, angles, center, oversample, p, sigma, lam);
                    T fy = error(y, sino, angles, center, oversample, p, sigma, lam);
                    T gy = 0.5 * step * g.norm();
                    if (fx > (fy + gy))
                        step *= 0.9;
                    else {
                        step_prev = step;
                        t = tnew;
                        xold = x;
                        break;
                    }
                }
                T e = error(x, sino, angles, center, oversample, p, sigma, lam);
                std::cout << "iter: " << iter << ", error: " << e << std::endl;
            }
            return x;
        }
    };

} // namespace tomocam

#endif // TOMOCAM_OPTIMIZE__H
