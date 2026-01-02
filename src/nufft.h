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

#include <iostream>

#include <cuda.h>

#include "common.h"
#include "dev_array.h"
#include "finufft_plan_cache.h"
#include "gpu/utils.cuh"
#include "grid.h"

#ifndef NUFFT_H
#define NUFFT_H

namespace tomocam::nufft {

    // 2-dimensional NUFFT from non-uniform to uniform grid (with caching)
    template <typename T>
    DeviceArray<cuda::std::complex<T>>
    nufft2d1_cache(DeviceArray<cuda::std::complex<T>> &c, const Grid<T> &nugrid,
                   dim3_t dims = {0, 0, 0}) {

        // check if device id is same as the one used for the grid
        int dev_id;
        SAFE_CALL(cudaGetDevice(&dev_id));
        if (dev_id != nugrid.dev_id()) {
            throw std::runtime_error("Device id mismatch");
        }

        // allocate return array
        if (dims.isNULL()) {
            dims = dim3_t{c.nslices(), nugrid.npixels(), nugrid.npixels()};
        }
        DeviceArray<cuda::std::complex<T>> fk(dims);
        std::array<int64_t, 2> n_modes = {
            static_cast<int64_t>(fk.dims().y),
            static_cast<int64_t>(fk.dims().z),
        };

        auto &plan = tomocam::nufft::plans::cache<T>.get_plan(1, 2, n_modes, 1);

        // set the non-uniform points
        plan.set_points(nugrid.size(), nugrid.x(), nugrid.y());

        // execute the plan
        plan.execute(c.dev_ptr(), fk.dev_ptr());
        return fk;
    }

    // 2-dimensional NUFFT from non-uniform to uniform grid
    template <typename T>
    DeviceArray<cuda::std::complex<T>>
    nufft2d1(DeviceArray<cuda::std::complex<T>> &c, const Grid<T> &nugrid,
             dim3_t dims = {0, 0, 0}) {

        // check if device id is same as the one used for the grid
        int dev_id;
        SAFE_CALL(cudaGetDevice(&dev_id));
        if (dev_id != nugrid.dev_id()) {
            throw std::runtime_error("Device id mismatch");
        }

        // allocate return array
        if (dims.isNULL()) {
            dims = dim3_t{c.nslices(), nugrid.npixels(), nugrid.npixels()};
        }
        DeviceArray<cuda::std::complex<T>> fk(dims);
        std::array<int64_t, 2> n_modes = {
            static_cast<int64_t>(fk.dims().y),
            static_cast<int64_t>(fk.dims().z),
        };

        FinufftPlanWrapper<T> plan;
        plan.make_plan(1, 2, n_modes.data(), 1);
        plan.set_points(nugrid.size(), nugrid.x(), nugrid.y());
        plan.execute(c.dev_ptr(), fk.dev_ptr());

        return fk;
    }

    // 2-dimensional NUFFT Uniform -> Non-uniform (with caching)
    template <typename T>
    DeviceArray<cuda::std::complex<T>>
    nufft2d2_cache(DeviceArray<cuda::std::complex<T>> &fk, const Grid<T> &nugrid) {

        // check if device id is same as the one used for the grid
        int dev_id;
        SAFE_CALL(cudaGetDevice(&dev_id));
        if (dev_id != nugrid.dev_id()) {
            std::cerr << "Device id mismatch" << std::endl;
            exit(1);
        }

        // allocate return array
        dim3_t dims = {fk.nslices(), nugrid.nprojs(), nugrid.npixels()};
        DeviceArray<cuda::std::complex<T>> c(dims);

        std::array<int64_t, 2> n_modes = {
            static_cast<int64_t>(fk.dims().y),
            static_cast<int64_t>(fk.dims().z),
        };

        auto &plan = tomocam::nufft::plans::cache<T>.get_plan(2, 2, n_modes, -1);

        // set the non-uniform points
        plan.set_points(nugrid.size(), nugrid.x(), nugrid.y());

        // execute the plan
        plan.execute(c.dev_ptr(), fk.dev_ptr());
        return c;
    }

    // 2-dimensional NUFFT Uniform -> Non-uniform
    template <typename T>
    DeviceArray<cuda::std::complex<T>>
    nufft2d2(DeviceArray<cuda::std::complex<T>> &fk, const Grid<T> &nugrid) {

        // check if device id is same as the one used for the grid
        int dev_id;
        SAFE_CALL(cudaGetDevice(&dev_id));
        if (dev_id != nugrid.dev_id()) {
            std::cerr << "Device id mismatch" << std::endl;
            exit(1);
        }

        // allocate return array
        dim3_t dims = {fk.nslices(), nugrid.nprojs(), nugrid.npixels()};
        DeviceArray<cuda::std::complex<T>> c(dims);

        std::array<int64_t, 2> n_modes = {
            static_cast<int64_t>(fk.dims().y),
            static_cast<int64_t>(fk.dims().z),
        };

        FinufftPlanWrapper<T> plan;
        plan.make_plan(2, 2, n_modes.data(), -1);
        plan.set_points(nugrid.size(), nugrid.x(), nugrid.y());
        plan.execute(c.dev_ptr(), fk.dev_ptr());
        return c;
    }
} // namespace tomocam::nufft

#endif // NUFFT_H
