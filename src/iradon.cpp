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
#include <omp.h>

#include "dist_array.h"
#include "internals.h"
#include "machine.h"
#include "shipper.h"
#include "types.h"

#include "dev_array.h"
#include "scheduler.h"

namespace tomocam {

    template <typename T>
    void backproject(Partition<T> sino, Partition<T> output,
        const std::vector<T> &angles, T center, int device) {

        // select device
        SAFE_CALL(cudaSetDevice(device));

        // create NUFFT Grid
        int nproj = static_cast<int>(angles.size());
        int ncols = sino.ncols();
        auto grid = NUFFT::Grid<T>(nproj, ncols, angles.data(), device);

        // create a data shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // input dimensions
        dim3_t idims = sino.dims();

        // subpartitions
        int nparts =
            Machine::config.num_of_partitions(output.dims(), output.bytes());
        auto sub_sinos = create_partitions(sino, nparts);
        auto sub_outputs = create_partitions(output, nparts);

        // start a scheduler
        Scheduler<Partition<T>, DeviceArray<T>> scheduler(sub_sinos);
        while(scheduler.has_work()) {
            auto task = scheduler.get_work();
            if (task.has_value()) {
                auto [i, d_sino] = task.value();
                auto d_recn = backproject(d_sino, grid, center);

                // copy the result to the output
                shipper.push(sub_outputs[i], d_recn);
            }
        }
    }

    // back projection
    template <typename T>
    DArray<T> backproject(DArray<T> &input, const std::vector<T> &angles,
        T center) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > input.nslices()) nDevice = input.nslices();

        // output dimensions
        dim3_t dims = {input.nslices(), input.ncols(), input.ncols()};
        DArray<T> output(dims);

        // create partitions
        auto p1 = create_partitions(input, nDevice);
        auto p2 = create_partitions(output, nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++)
            backproject(p1[i], p2[i], angles, center, i);

        // wait for devices to finish
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
        }
        return output;
    }

    // explicit instantiation
    template DArray<float> backproject(DArray<float> &,
        const std::vector<float> &, float);
    template DArray<double> backproject(DArray<double> &,
        const std::vector<double> &, double);

} // namespace tomocam
