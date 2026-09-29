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
#ifndef NUFFT_H
#define NUFFT_H

#include <cuda.h>
#include <iostream>

#include "common.h"
#include "dev_array.h"
#include "finufft_plan_cache.h"
#include "gpu/utils.cuh"
#include "grid.h"

namespace tomocam::nufft {

    // 2-dimensional NUFFT from non-uniform to uniform grid
    template <typename T>
    void nufft2d1(DeviceArray<cuda::std::complex<T>> &c,
                  DeviceArray<cuda::std::complex<T>> &f, const Grid<T> &pg,
                  bool use_cache = false) {

        // ensure device matches grid's device
        DeviceGuard guard(pg.dev_id());

        std::array<int64_t, 2> n_modes = {
            static_cast<int64_t>(f.dims().y),
            static_cast<int64_t>(f.dims().z),
        };
        int ntrans = static_cast<int>(f.dims().x);

        if (use_cache) {
            auto &plan = tomocam::nufft::plans::cache<T>.get_plan(
                1, 2, n_modes, 1, ntrans, pg.dev_id());
            plan.set_points(pg.size(), pg.x(), pg.y());
            plan.execute(c.dev_ptr(), f.dev_ptr());
        } else {
            FinufftPlanWrapper<T> plan;
            plan.make_plan(1, 2, n_modes, 1, ntrans, pg.dev_id());
            plan.set_points(pg.size(), pg.x(), pg.y());
            plan.execute(c.dev_ptr(), f.dev_ptr());
        }
    }

    // 2-dimensional NUFFT Uniform -> Non-uniform
    template <typename T>
    void nufft2d2(DeviceArray<cuda::std::complex<T>> &c,
                  DeviceArray<cuda::std::complex<T>> &f, const Grid<T> &pg,
                  bool use_cache = false) {

        // ensure device matches grid's device
        DeviceGuard guard(pg.dev_id());

        std::array<int64_t, 2> n_modes = {
            static_cast<int64_t>(f.dims().y),
            static_cast<int64_t>(f.dims().z),
        };
        int ntrans = static_cast<int>(f.dims().x);

        if (use_cache) {
            auto &plan = tomocam::nufft::plans::cache<T>.get_plan(
                2, 2, n_modes, -1, ntrans, pg.dev_id());
            plan.set_points(pg.size(), pg.x(), pg.y());
            plan.execute(c.dev_ptr(), f.dev_ptr());
        } else {
            FinufftPlanWrapper<T> plan;
            plan.make_plan(2, 2, n_modes, -1, ntrans, pg.dev_id());
            plan.set_points(pg.size(), pg.x(), pg.y());
            plan.execute(c.dev_ptr(), f.dev_ptr());
        }
    }
} // namespace tomocam::nufft

#endif // NUFFT_H
