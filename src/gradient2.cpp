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
#include "toeplitz.h"
#include "types.h"

namespace tomocam {

    template <typename T>
    void gradient2_(Partition<T> f, Partition<T> sinoT, Partition<T> df,
        const PointSpreadFunction<T> &psf, int device_id) {

        // set device
        SAFE_CALL(cudaSetDevice(device_id));

        // sub-partitions
        int nparts = Machine::config.num_of_partitions(sinoT.dims(), sinoT.bytes());
        auto p1 = create_partitions(f, nparts);
        auto p2 = create_partitions(sinoT, nparts);
        auto p3 = create_partitions(df, nparts);

        // create a data shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // creater a scheduler, and assign work
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> s(p1, p2);

        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto [idx, d_f, d_sinoT] = work.value();

                // compute gradient
                auto d_g = DeviceArray<T>();
                if (d_f.nslices() == psf.batch_size()) {
                    // partition size matches the batch size
                    d_g = psf.convolve2(d_f) - d_sinoT;
                } else {
                    // otherwise, use the slow method
                    d_g = psf.convolve(d_f) - d_sinoT;
                }

                // copy gradient to host
                shipper.push(p3[idx], d_g);
            }
        }
    }

    // Multi-GPU calll
    template <typename T>
    DArray<T> gradient2(DArray<T> &solution, DArray<T> &sinoT,
        const std::vector<PointSpreadFunction<T>> &psfs) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > sinoT.nslices()) nDevice = sinoT.nslices();

        // allocate memory for gradient
        DArray<T> gradient(solution.dims());

        auto p1 = create_partitions(solution, nDevice);
        auto p2 = create_partitions(sinoT, nDevice);
        auto p3 = create_partitions(gradient, nDevice);

        // launch all the available devices
        std::vector<std::thread> threads(nDevice);
        for (int i = 0; i < nDevice; i++) {
            threads[i] = std::thread(gradient2_<T>, p1[i], p2[i], p3[i], std::cref(psfs[i]), i);
        }

        // wait for devices to finish
        for (int i = 0; i < nDevice; i++) {
            threads[i].join();
        }
        return gradient;
    }

    // specialization for float
    template DArray<float> gradient2<float>(DArray<float> &solution,
        DArray<float> &sinoT,
        const std::vector<PointSpreadFunction<float>> &psfs);
    // specialization for double
    template DArray<double> gradient2<double>(DArray<double> &solution,
        DArray<double> &sinoT,
        const std::vector<PointSpreadFunction<double>> &psfs);

} // namespace tomocam
