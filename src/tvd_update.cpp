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
#include "types.h"

#include "gpu/totalvar.cuh"

#include "debug.h"

namespace tomocam {

    template <typename T>
    void total_var(Partition<T> sol, Partition<T> grad, float p, float sigma,
        int device) {

        // initalize the device
        SAFE_CALL(cudaSetDevice(device));

        // create stream to copy data to host
        cudaStream_t work_s;
        cudaStream_t out_s;
        SAFE_CALL(cudaStreamCreate(&out_s));

        // create sub-partitions with halo
        int nslcs = Machine::config.slicesPerStream();
        int nparts = sol.nslices() / nslcs;
        if (sol.nslices() % nslcs != 0) nparts++;
        auto sub_sols = create_partitions(sol, nparts, 1);
        auto sub_grads = create_partitions(grad, nparts);

        if (sub_sols.size() != sub_grads.size()) {
            throw std::runtime_error(
                "Error: sub_sols and sub_grads have different sizes");
        }

        // create scheduler
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> scheduler(
            sub_sols, sub_grads);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {

                // unpack the data
                auto [idx, d_s, d_g] = work.value();

                // update the total variation
                gpu::add_total_var<T>(d_s, d_g, p, sigma, work_s);
                SAFE_CALL(cudaStreamSynchronize(work_s));
                d_g.copy_to(sub_grads[idx], out_s);
            }
        }
        SAFE_CALL(cudaStreamSynchronize(out_s));
        SAFE_CALL(cudaStreamDestroy(out_s));
    }

    // multi-GPU call
    template <typename T>
    void add_total_var(DArray<T> &sol, DArray<T> &grad, float p, float sigma) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > sol.nslices()) nDevice = sol.nslices();

        auto p1 = create_partitions(sol, nDevice, 1);
        auto p2 = create_partitions(grad, nDevice);

        // #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++){
            total_var<T>(p1[i], p2[i], p, sigma, i);
        }

        // #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
        }
    }

    // explicit instantiation
    template void add_total_var<float>(DArray<float> &, DArray<float> &, float,
        float);
    template void add_total_var<double>(DArray<double> &, DArray<double> &,
        float, float);

} // namespace tomocam
