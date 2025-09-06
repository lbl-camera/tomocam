/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <thread>
#include <vector>

#include "common.h"
#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "partition.h"
#include "scheduler.h"
#include "shipper.h"

#include "gpu/padding.cuh"
#include "gpu/utils.cuh"

namespace tomocam {

    template <typename T>
    void pad2d_(Partition<T> arr, Partition<T> arr2, int npad, PadType type, int device) {

        // set the device
        SAFE_CALL(cudaSetDevice(device));

        // create subpartitions
        int nparts =
            Machine::config.num_of_partitions(arr.dims(), arr.bytes());
        auto p1 = create_partitions<T>(arr, nparts);
        auto p2 = create_partitions<T>(arr2, nparts);

        // data shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // start GPU Scheduler
        Scheduler<Partition<T>, DeviceArray<T>> scheduler(p1);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto[idx, d_arr] = work.value();

                // pad the sinogram
                auto d_arr2 = gpu::pad2d(d_arr, npad, PadType::SYMMETRIC);

                // copy data to partition
                shipper.push(p2[idx], d_arr2);
            }
        }
    }

    template <typename T>
    DArray<T> pad2d(DArray<T> &arr, int npad, PadType type) {

        int ndevices = Machine::config.num_of_gpus();
        if (arr.nslices() < ndevices) { ndevices = 1; }

        // allocate memory for the padded array
        int nslices = arr.nslices();
        int nrows = arr.nrows() + npad;
        int ncols = arr.ncols() + npad;
        DArray<T> arr2({nslices, nrows, ncols});

        auto p1 = create_partitions<T>(arr, ndevices);
        auto p2 = create_partitions<T>(arr2, ndevices);

        std::vector<std::thread> threads(ndevices);
        for (int i = 0; i < ndevices; i++) {
            threads[i] = std::thread(pad2d_<T>, p1[i], p2[i], npad, type, i);
        }
        Machine::config.barrier();
        for (auto &t : threads) { t.join(); }
        
        return arr2;
    }

    // explicit instantiation
    template DArray<float> pad2d(DArray<float> &, int, PadType);
    template DArray<double> pad2d(DArray<double> &, int, PadType);

} // namespace tomocam
