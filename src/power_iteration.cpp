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

#include <cmath>
#include <functional>

#include "dist_array.h"
#include "dist_array_ops.h"
#include "optimize.h"

namespace tomocam {

    template <typename T>
    T power_iteration(std::function<DArray<T>(DArray<T> &)> A, dim3_t dims,
                      int max_iters, T tol) {

        // start from a unit-norm all-ones array
        DArray<T> v(dims);
        v.init(1);
        v = v / std::sqrt(v.norm());

        T lambda = 0;
        for (int iter = 0; iter < max_iters; ++iter) {
            DArray<T> Av = A(v);

            // Rayleigh quotient: v^T A v (v is unit-norm)
            T lambda_new = array::dot(v, Av);

            T mag = std::sqrt(Av.norm());
            if (mag < static_cast<T>(1e-12)) return 0;
            v = Av / mag;

            if (iter > 0 &&
                std::abs(lambda_new - lambda) <= tol * std::abs(lambda_new)) {
                return lambda_new;
            }
            lambda = lambda_new;
        }
        return lambda;
    }

    // explicit instantiation
    template float power_iteration(std::function<DArray<float>(DArray<float> &)>,
                                   dim3_t, int, float);
    template double power_iteration(std::function<DArray<double>(DArray<double> &)>,
                                    dim3_t, int, double);

} // namespace tomocam
