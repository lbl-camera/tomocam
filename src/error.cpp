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

#include "dev_array.h"
#include "dist_array.h"
#include "internals.h"
#include "machine.h"
#include "scheduler.h"
#include "toeplitz.h"
#include "types.h"

namespace tomocam {

    template <typename T>
    T funcval(Partition<T> recon, Partition<T> sino,
        const NUFFT::Grid<T> &nugrid, int offset, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // sub-partitions
        int nslcs = Machine::config.num_of_partitions(recon.nslices());
        auto p1 = create_partitions(recon, nslcs);
        auto p2 = create_partitions(sino, nslcs);
        T sum = 0;

        // create a scheduler
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> scheduler(p1,
            p2);

        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto [idx, d_recon, d_sino] = work.value();
                auto t1 = project(d_recon, nugrid, offset);
                auto t2 = t1 - d_sino;
                sum += t2.dot(t2);
            }
        }
        return sum;
    }

    // Multi-GPU calll
    template <typename T>
    T function_value(DArray<T> &recon, DArray<T> &sino,
        const std::vector<NUFFT::Grid<T>> &nugrids, int center) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > recon.nslices()) nDevice = recon.nslices();

        // center offset
        int offset = center - sino.nslices() / 2;

        auto p1 = create_partitions(recon, nDevice);
        auto p2 = create_partitions(sino, nDevice);

        std::vector<T> retval(nDevice);
        #pragma omp parallel for
        for (int i = 0; i < nDevice; i++)
            retval[i] = funcval(p1[i], p2[i], nugrids[i], offset, i);

        // wait for devices to finish
        T fval = 0;
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
            fval += retval[i];
        }
        return (fval / recon.size());
    }

    // explicit instantiation
    template float function_value(DArray<float> &recon, DArray<float> &sino,
        const std::vector<NUFFT::Grid<float>> &nugrids, int center);
    template double function_value(DArray<double> &recon, DArray<double> &sino,
        const std::vector<NUFFT::Grid<double>> &nugrids, int center);
} // namespace tomocam
