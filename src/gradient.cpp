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
#include <thread>
#include <vector>

#include "dev_array.h"
#include "dist_array.h"
#include "internals.h"
#include "machine.h"
#include "scheduler.h"
#include "shipper.h"
#include "types.h"

namespace tomocam {

    template <typename T>
    void gradient_(Partition<T> f, Partition<T> sinoT, Partition<T> df,
                   const nufft::Grid<T> &nugrid, int device_id) {

        // set device
        SAFE_CALL(cudaSetDevice(device_id));

        // sub-partitions
        int nparts = Machine::config.num_of_partitions(sinoT.dims(), sinoT.bytes());
        auto p1 = create_partitions(f, nparts);
        auto p2 = create_partitions(sinoT, nparts);
        auto p3 = create_partitions(df, nparts);

        // normalization factor
        auto scale = static_cast<T>(std::pow(f.ncols(), 3));

        // create a shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // creater a scheduler, and assign work
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> s(p1, p2);

        // cache FINUFFT plans, and reuse them
        bool cache_plans = true;
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto &&[idx, d_f, d_sinoT] = std::move(work.value());

                // allocate intermediate array
                dim3_t proj_dims = {d_f.nslices(), nugrid.nprojs(),
                                    nugrid.npixels()};
                auto temp = DeviceArray<cuda::std::complex<T>>(proj_dims);
                auto d_gcmplx = DeviceArray<gpu::complex_t<T>>(d_f.dims());

                auto d_fcmplx = to_complex<T>(d_f);
                // call NUFFT operations, set cache_plans to true
                nufft::nufft2d2(temp, d_fcmplx, nugrid, cache_plans);
                nufft::nufft2d1(temp, d_gcmplx, nugrid, cache_plans);
                auto d_g = to_real<T>(d_gcmplx);
                d_g = (d_g - d_sinoT) / scale;

                // copy gradient to host
                shipper.push(p3[idx], std::move(d_g));
            }
        }
    }

    // Multi-GPU calll
    template <typename T>
    DArray<T> gradient(DArray<T> &solution, DArray<T> &sinoT,
                       const std::vector<nufft::Grid<T>> &nugrids) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > sinoT.nslices()) nDevice = sinoT.nslices();

        // allocate memory for gradient
        DArray<T> gradient(solution.dims());

        auto p1 = create_partitions(solution, nDevice);
        auto p2 = create_partitions(sinoT, nDevice);
        auto p3 = create_partitions(gradient, nDevice);

        // create a vector std::threads to launch the gradient function
        std::vector<std::thread> threads(nDevice);
        for (int i = 0; i < nDevice; i++) {
            threads[i] = std::thread(gradient_<T>, p1[i], p2[i], p3[i],
                                     std::cref(nugrids[i]), i);
        }

        // waht for all threads to join
        for (auto &t : threads) { t.join(); }

        // return the gradient
        return gradient;
    }

    // Explicit instantiation
    template DArray<float> gradient(DArray<float> &, DArray<float> &,
                                    const std::vector<nufft::Grid<float>> &);
    template DArray<double> gradient(DArray<double> &, DArray<double> &,
                                     const std::vector<nufft::Grid<double>> &);

} // namespace tomocam
