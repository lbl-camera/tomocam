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

#include "common.h"
#include "dev_array.h"
#include "dist_array.h"
#include "gpu/utils.cuh"
#include "machine.h"
#include "partition.h"
#include "scheduler.h"
#include "shipper.h"

#include "gpu/padding.cuh"

namespace tomocam {

    template <typename T>
    void postproc(Partition<T> soln, Partition<T> soln2, int npad, int device) {

        // set the device
        SAFE_CALL(cudaSetDevice(device));

        // create subpartitions
        int nparts =
            Machine::config.num_of_partitions(soln.dims(), soln.bytes());
        auto p1 = create_partitions<T>(soln, nparts);
        auto p2 = create_partitions<T>(soln2, nparts);

        // data shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // start GPU Scheduler
        Scheduler<Partition<T>, DeviceArray<T>> scheduler(p1);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto [idx, d_soln] = work.value();

                // remove padding
                auto d_soln2 = gpu::unpad2d(d_soln, 2 * npad);

                // copy data to partition
                shipper.push(p2[idx], d_soln2);
            }
        }
    }

    template <typename T>
    DArray<T> postproc(DArray<T> &soln, int npixels) {

        int ndevices = Machine::config.num_of_gpus();
        if (soln.nslices() < ndevices) { ndevices = 1; }

        // calculate the dimensions of cropped image
        int npad = (soln.ncols() - npixels) / 2;

        // dimensions of the cropped reconstruction
        int nslcs = soln.nslices();
        int nrows = soln.nrows() - 2 * npad;
        int ncols = soln.ncols() - 2 * npad;

        // allocate memory for cropped solution
        DArray<T> soln2(dim3_t(nslcs, nrows, ncols));

        // create a partition per device
        auto p1 = create_partitions<T>(soln, ndevices);
        auto p2 = create_partitions<T>(soln2, ndevices);

        #pragma omp parallel for
        for (int dev = 0; dev < ndevices; dev++) {
            postproc(p1[dev], p2[dev], npad, dev);
        }

        // synchronize all devices
        #pragma omp parallel for
        for (int dev = 0; dev < ndevices; dev++) {
            SAFE_CALL(cudaSetDevice(dev));
            SAFE_CALL(cudaDeviceSynchronize());
        }
        return soln2;
    }

    // explicit instantiation
    template DArray<float> postproc(DArray<float> &, int);
    template DArray<double> postproc(DArray<double> &, int);

} // namespace tomocam
