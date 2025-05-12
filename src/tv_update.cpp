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

#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>

#include "dev_array.h"
#include "dist_array.h"
#include "internals.h"
#include "machine.h"
#include "scheduler.h"
#include "shipper.h"
#include "types.h"

#ifdef MULTIPROC
#include "multiproc.h"
#endif

#include "gpu/totalvar.cuh"

namespace tomocam {

    template <typename T>
    void total_var2(Partition<T> sol, Partition<T> grad, T sigma, T p, int device) {

        // initalize the device
        SAFE_CALL(cudaSetDevice(device));

        // create sub-partitions with halo
        auto nparts =Machine::config.num_of_partitions(grad.dims(), grad.bytes());
        auto sub_sols = create_partitions(sol, nparts, 1);
        auto sub_grads = create_partitions(grad, nparts);

        // create a shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // create scheduler
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> scheduler(
            sub_sols, sub_grads);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {

                // unpack the data
                auto[idx, d_s, d_g] = work.value();

                // update the total variation
                gpu::add_total_var2<T>(d_s, d_g, sigma, p);

                // d_g.copy_to(sub_grads[idx], out_s);
                shipper.push(sub_grads[idx], d_g);
            }
        }
    }

    // multi-GPU call
    template <typename T>
    void add_total_var2(DArray<T> &sol, DArray<T> &grad, T sigma, T p) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > sol.nslices()) nDevice = sol.nslices();

        dim3_t dims = sol.dims();
        #ifdef MULTIPROC
        int myrank = multiproc::mp.myrank();
        int size = multiproc::mp.nprocs();
        if (myrank > 0) dims.x += 1;
        if (myrank < size - 1) dims.x += 1;
        int start = myrank == 0 ? 0 : 1;

        // allcate memory for halo
        DArray<T> sol2(dims);
        std::copy(sol.begin(), sol.end(), sol2.slice(start));
        sol2.update_neigh_proc();
        auto p1 = create_partitions(sol2, nDevice, 1);
        #else
        auto p1 = create_partitions(sol, nDevice, 1);
        #endif

        auto p2 = create_partitions(grad, nDevice);

        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            total_var2<T>(p1[i], p2[i], sigma, p, i);
        }

    }

    // explicit instantiation
    template void add_total_var2<float>(DArray<float> &, DArray<float> &, float, float);
    template void add_total_var2<double>(DArray<double> &, DArray<double> &, double, double);

} // namespace tomocam
