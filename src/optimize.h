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
#include <functional>

#include "dist_array.h"
#include "machine.h"
#include "tomocam.h"

#ifdef DEBUG
#include "debug.h"
#endif

#ifndef TOMOCAM_OPTIMIZE__H
#define TOMOCAM_OPTIMIZE__H

namespace tomocam {
    // abstract preconditioner interface for cgsolver
    template <typename T>
    class Precond {
      public:
        virtual ~Precond() = default;
        virtual DArray<T> apply(DArray<T> &r) = 0;
    };

    // identity preconditioner - the default when the caller doesn't
    // supply a real one
    template <typename T>
    class IPrecond : public Precond<T> {
      public:
        DArray<T> apply(DArray<T> &r) override { return r; }
    };

    template <typename T>
    DArray<T> cgsolver(std::function<DArray<T>(DArray<T> &)> A, const DArray<T> &b,
                       const DArray<T> &x0, const ReconParams &params,
                       Precond<T> *precond = nullptr);

    // FISTA-style solver with backtracking line search and step-size reset
    template <typename T>
    DArray<T> nagopt(std::function<DArray<T>(DArray<T> &)> gradient,
                     std::function<T(DArray<T> &)> loss, const DArray<T> &x0,
                     T step_size, const ReconParams &params);

    // split-Bregman TV-regularized solver: minimizes
    //   argmin_x  ||A x - b||^2 + lambda * TV(x)
    // via alternating CG x-update and shrinkage d/b-updates.
    template <typename T>
    DArray<T> split_bregman(std::function<DArray<T>(DArray<T> &)> A,
                            const DArray<T> &b, const DArray<T> &x0,
                            const ReconParams &params, Precond<T> *precond = nullptr);

} // namespace tomocam

#endif // TOMOCAM_OPTIMIZE__H
