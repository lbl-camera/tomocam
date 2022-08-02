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

const float inf = std::numeric_limits<float>::infinity();

#ifndef TOMOCAM_OPTIMIZE__H
#define TOMOCAM_OPTIMIZE__H

namespace tomocam {

    template <typename T>
    inline DArray<T> calc_gradient(DArray<T> &x, DArray<T> &sino,
            float * angles, float center, float oversample, 
            float p, float sigma) {
        DArray g = x;
        gradient(x, sino, angles, center, oversample);
        add_total_var(x, g, p, sigma);
        return g;
    }

    template <typename T>
    inline T calc_lipschitz(DArray<T> &x, DArray<T> &sino,
            float * angles, float center, float oversample, 
            float p, float sigma) {
        DArray g = x;
        gradient(x, sino, angles, center, oversample);
        add_total_var(x, g, p, sigma);
        add_tv_hessian(g, sigma);
        return (2 * g.max());
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
                float oversample, float p, float sigma) {

            // initialize 
            DArray<T> x(dims_);
            x.init(1.);
            DArray<T> xold(dims_);
            xold.init(1.);
            DArray<T> y(dims_);

            T t = 1;
            T tnew = 1; 
            // compute Lipschitz
            T lipschitz_ = calc_lipschitz(xold, sino, angles, center, oversample, p, sigma);

            // gradient step size
            T step = 1./lipschitz_;

            for (int iter = 0; iter < max_iters_; iter++) {
                float beta = tnew * (1./t - 1);
                y = x + (x - xold) * beta;
                auto g = calc_gradient(y, sino, angles, center, oversample, p, sigma);
                xold = x;
                x = y -  g * step;
                auto e = g.norm() / static_cast<float>(g.size());
            
                // update theta
                float temp = 0.5 * (std::sqrt(std::pow(t,4) 
                            + 4 * std::pow(t,2))
                        - std::pow(t,2));
                t = tnew;
                tnew = temp;

                std::cout << "tnew = " << tnew << std::endl;
                std::cout << "iter: " << iter << ", error: " << e << std::endl;
            }
            return x;
        }
    };

} // namespace tomocam

#endif // TOMOCAM_OPTIMIZE__H
