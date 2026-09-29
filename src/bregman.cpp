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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <utility>

#include "dist_array.h"
#include "optimize.h"
#include "tomocam.h"

#include "timer.h"

#include "dist_array_ops.h"

namespace tomocam {

    constexpr double EPSILON = 1e-8;

    template <typename T>
    DArray<T> split_bregman(std::function<DArray<T>(DArray<T> &)> A,
                            const DArray<T> &b, const DArray<T> &x0,
                            const ReconParams &params, Precond<T> *precond) {

        // if no preconditioner is provided, use the identity
        IPrecond<T> identity;
        if (!precond) precond = &identity;

        T mu = static_cast<T>(params.mu);
        T lambda_mu = static_cast<T>(params.lambda) / mu;

        DArray<T> x = x0;
        DArray<T> x_old = x0;

        // Bregman split variables: d (auxiliary TV variable) and bregman_b
        // (Bregman update accumulator). Named bregman_b, not b, to avoid
        // shadowing the function parameter `b` (the CG right-hand side).
        // DArray<T> has no default constructor, so std::array elements must
        // be aggregate-initialized rather than default-constructed + assigned.
        dim3_t dims = x.dims();
        auto zeros = [&dims]() {
            DArray<T> a(dims);
            a.init(T(0));
            return a;
        };
        std::array<DArray<T>, 3> d{zeros(), zeros(), zeros()};
        std::array<DArray<T>, 3> bregman_b{zeros(), zeros(), zeros()};

        // update A^TA to add laplacian of x
        // Ap  = (A^TA  +   mu * (-laplacian)) u   [laplacian = div(grad(u))]
        // cgsolver calls A(x)/A(p) with a non-const DArray<T>&, so the
        // functor's parameter must be non-const to match. Computed via
        // array::axpy (GPU-resident) instead of DArray's host operator-/*,
        // same fix as nagopt.cpp's FISTA loop.
        std::function<DArray<T>(DArray<T> &)> Ap = [&](DArray<T> &u) -> DArray<T> {
            auto Au = A(u);
            auto lap = array::laplacian(u);
            // lap is a dead local after this call -- move it in so axpy's
            // by-value first arg is move-constructed, not a full host copy
            return array::axpy(std::move(lap), -mu, Au); // -mu*lap + Au
        };

        for (size_t iter = 0; iter < params.max_iters; ++iter) {

            // x-update: solve (A^TA + mu*(-laplacian))x = b - mu*divergence(d -
            // bregman_b)
            //
            // Derivation: with eta=mu the penalty weight on the constraint
            // d = grad_u(x), the x-subproblem's stationarity condition is
            // (A^TA + mu*grad_u^T*grad_u) x = b + mu*grad_u^T*(d - bregman_b).
            // grad_u^T = -divergence (confirmed empirically by test/finitdiff.cpp's
            // adjoint check: <grad_u(x),y> == -<x,divergence(y)> up to a
            // boundary term), so grad_u^T*grad_u = -laplacian and
            // grad_u^T*(d-bregman_b) = -divergence(d-bregman_b), giving the
            // signs used here and in the Ap operator below.
            // d_b = d - bregman_b and rhs = b - mu*divergence(d_b), via
            // array::axpy (GPU-resident) instead of DArray's host operator-/*
            std::array<DArray<T>, 3> d_b{
                array::axpy(bregman_b[0], static_cast<T>(-1), d[0]),
                array::axpy(bregman_b[1], static_cast<T>(-1), d[1]),
                array::axpy(bregman_b[2], static_cast<T>(-1), d[2])};
            auto div_db = array::divergence(d_b);
            // div_db is dead after this -- move, not copy
            DArray<T> rhs = array::axpy(std::move(div_db), -mu, b); // -mu*div_db + b

            // use conjugate gradient to solve the linear system
            x = cgsolver<T>(Ap, rhs, x, params, precond);

            // isotropic TV shrinkage + Bregman update, fully on GPU
            auto dx = array::grad_u(x);
            array::shrinkage(dx, bregman_b, d, lambda_mu, static_cast<T>(EPSILON));

            // TODO: fuse the following two kernels
            auto x_diff = array::axpy(std::move(x_old), static_cast<T>(-1), x);
            T norm_diff = array::dot(x_diff, x_diff); // squared L2 norm

            fprintf(stdout, "split_bregman iter: %zu, norm_diff: %.4e\n", iter,
                    static_cast<double>(norm_diff));
            if (norm_diff < static_cast<T>(params.xtol * params.xtol)) {
                fprintf(stdout, "split_bregman converged in %zu iterations\n",
                        iter + 1);
                break;
            }
            x_old = x;
        }
        return x;
    }

    // explicit instantiation
    template DArray<float>
    split_bregman(std::function<DArray<float>(DArray<float> &)>,
                  const DArray<float> &, const DArray<float> &, const ReconParams &,
                  Precond<float> *);
    template DArray<double>
    split_bregman(std::function<DArray<double>(DArray<double> &)>,
                  const DArray<double> &, const DArray<double> &, const ReconParams &,
                  Precond<double> *);

} // namespace tomocam
