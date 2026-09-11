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
#include <utility>

#include "dist_array.h"
#include "optimize.h"
#include "tomocam.h"

#include "dist_array_ops.h"

namespace tomocam {

    template <typename T>
    DArray<T> cgsolver(std::function<DArray<T>(DArray<T> &)> A, const DArray<T> &b,
                       const DArray<T> &x0, const ReconParams &params,
                       Precond<T> *precond) {

        IPrecond<T> identity;
        if (!precond) precond = &identity;

        DArray<T> x = x0;
        DArray<T> r = array::axpy(A(x), static_cast<T>(-1), b); // r = b - A*x
        DArray<T> z = precond->apply(r);
        DArray<T> p = z;
        T rho = array::dot(r, z);

        for (size_t iter = 0; iter < params.inner_iters; ++iter) {

            DArray<T> Ap = A(p);
            T pAp = array::dot(p, Ap);
            if (std::abs(pAp) < static_cast<T>(1.e-10)) {
                fprintf(stdout, "CG solver failed to converge in %zu iterations\n",
                        iter);
                break;
            }
            T alpha = rho / pAp;
            array::xpay(x, alpha, p);   // x += alpha * p
            array::xpay(r, -alpha, Ap); // r -= alpha * A*p

            T r_norm2 = array::dot(r, r);
            if (r_norm2 < params.tol * params.tol) {
                fprintf(stdout, "CG solver converged in %zu iterations\n", iter + 1);
                break;
            }

            z = precond->apply(r);
            T rho_new = array::dot(r, z);
            T beta = rho_new / rho;
            array::xpay(z, beta, p); // z = z + beta * p  (new search direction)
            p = std::move(z);
            rho = rho_new;
        }
        return x;
    }

    // explicit instantiation
    template DArray<float> cgsolver(std::function<DArray<float>(DArray<float> &)>,
                                    const DArray<float> &, const DArray<float> &,
                                    const ReconParams &, Precond<float> *);
    template DArray<double> cgsolver(std::function<DArray<double>(DArray<double> &)>,
                                     const DArray<double> &, const DArray<double> &,
                                     const ReconParams &, Precond<double> *);

} // namespace tomocam
