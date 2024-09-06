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
    T funcval2(Partition<T> recon, Partition<T> sinoT,
        const PointSpreadFunction<T> &psf, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // sub-partitions
        int nslcs = Machine::config.num_of_partitions(recon.nslices());
        auto p1 = create_partitions(recon, nslcs);
        auto p2 = create_partitions(sinoT, nslcs);
        T sum = 0;

        // create a scheduler
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> scheduler(p1,
            p2);

        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto [idx, d_recon, d_sinoT] = work.value();
                auto t1 = psf.convolve(d_recon);
                auto t2 = d_recon.dot(t1);
                auto t3 = d_recon.dot(d_sinoT);
                sum += (t2 - 2 * t3);
            }
        }
        return sum / recon.ncols();
    }

    // Multi-GPU calll
    template <typename T>
    T function_value2(DArray<T> &recon, DArray<T> &sinoT,
        const std::vector<PointSpreadFunction<T>> &psf, T sino_sq) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > recon.nslices()) nDevice = recon.nslices();

        auto p1 = create_partitions(recon, nDevice);
        auto p2 = create_partitions(sinoT, nDevice);

        std::vector<T> retval(nDevice);
        // #pragma omp parallel for
        for (int i = 0; i < nDevice; i++)
            retval[i] = funcval2(p1[i], p2[i], psf[i], i);

        // wait for devices to finish
        T fval = sino_sq;
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
            fval += retval[i];
        }
        return (fval / recon.size());
    }

    // explicit instantiation
    template float function_value2(DArray<float> &, DArray<float> &,
        const std::vector<PointSpreadFunction<float>> &, float);
    template double function_value2(DArray<double> &, DArray<double> &,
        const std::vector<PointSpreadFunction<double>> &, double);
} // namespace tomocam
