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
    void gradient_(Partition<T> f, Partition<T> sino, Partition<T> df,
        const NUFFT::Grid<T> &nugrid, int offset, int device_id) {

        // set device
        SAFE_CALL(cudaSetDevice(device_id));

        // sub-partitions
        int nparts = Machine::config.num_of_partitions(sino.nslices());
        auto p1 = create_partitions(f, nparts);
        auto p2 = create_partitions(sino, nparts);
        auto p3 = create_partitions(df, nparts);

        // create a shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // creater a scheduler, and assign work
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> s(p1, p2);
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto [idx, d_f, d_sino] = work.value();

                auto t1 = backproject(d_f, nugrid, offset);
                auto t2 = t1 - d_sino;
                auto d_g = project(t2, nugrid, offset);

                // copy gradient to host
                shipper.push(p3[idx], d_g);
            }
        }
    }

    // Multi-GPU calll
    template <typename T>
    DArray<T> gradient(DArray<T> &solution, DArray<T> &sino,
        const std::vector<NUFFT::Grid<T>> &nugrids, int center) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > sino.nslices()) nDevice = sino.nslices();

        // offset
        int offset = center - sino.ncols() / 2;

        // allocate memory for gradient
        DArray<T> gradient(solution.dims());

        auto p1 = create_partitions(solution, nDevice);
        auto p2 = create_partitions(sino, nDevice);
        auto p3 = create_partitions(gradient, nDevice);

        // vecor to store partial function values
        std::vector<T> pfunc(nDevice, 0);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            gradient_<T>(p1[i], p2[i], p3[i], nugrids[i], offset, i);
        }

        // wait for devices to finish
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
        }

        return gradient;
    }

    // Explicit instantiation
    template DArray<float> gradient<float>(DArray<float> &, DArray<float> &,
        const std::vector<NUFFT::Grid<float>> &, int);
    template DArray<double> gradient<double>(DArray<double> &, DArray<double> &,
        const std::vector<NUFFT::Grid<double>> &, int);

} // namespace tomocam
