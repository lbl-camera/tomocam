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
#include "machine.h"
#include "partition.h"
#include "scheduler.h"
#include "shipper.h"

#include "gpu/fftshift.cuh"
#include "gpu/padding.cuh"
#include "gpu/utils.cuh"

namespace tomocam {

    template <typename T>
    void preproc(Partition<T> sino, Partition<T> sino2, int npad, int offset,
        int device) {

        // set the device
        SAFE_CALL(cudaSetDevice(device));

        // create subpartitions
        int nparts =
            Machine::config.num_of_partitions(sino2.dims(), sino2.bytes());
        auto p1 = create_partitions<T>(sino, nparts);
        auto p2 = create_partitions<T>(sino2, nparts);

        // data shipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // start GPU Scheduler
        Scheduler<Partition<T>, DeviceArray<T>> scheduler(p1);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto [idx, d_sino] = work.value();

                // pad the sinogram
                auto d_sino2 = gpu::pad1d(d_sino, 2 * npad, PadType::SYMMETRIC);

                // shift by twice the center offset
                d_sino2 = gpu::roll(d_sino2, offset);

                // copy data to partition
                shipper.push(p2[idx], d_sino2);
            }
        }
    }

    template <typename T>
    DArray<T> preproc(DArray<T> &sino, T center) {

        int ndevices = Machine::config.num_of_gpus();
        if (sino.nslices() < ndevices) { ndevices = 1; }

        // center shift
        int cen = static_cast<int>(std::round(center));
        int cen_offset = sino.ncols() / 2 - cen;

        // calculate the number of padding pixels ( ≥ √2  * sino.ncols())
        int npad = static_cast<int>(0.42 * sino.ncols()) / 2;
        if (std::abs(cen_offset) > npad) npad = std::abs(cen_offset);

        // dimensions of the padded sinogram
        int nslcs = sino.nslices();
        int nproj = sino.nrows();
        int ncols = sino.ncols() + 2 * npad;

        // create a new sinogram with the padded dimensions
        DArray<T> sino2(dim3_t(nslcs, nproj, ncols));

        // create a partition per device
        auto p1 = create_partitions<T>(sino, ndevices);
        auto p2 = create_partitions<T>(sino2, ndevices);

        #pragma omp parallel for
        for (int dev = 0; dev < ndevices; dev++) {
            preproc(p1[dev], p2[dev], npad, cen_offset, dev);
        }

        // synchronize all devices
        #pragma omp parallel for
        for (int dev = 0; dev < ndevices; dev++) {
            SAFE_CALL(cudaSetDevice(dev));
            SAFE_CALL(cudaDeviceSynchronize());
        }
        return sino2;
    }

    // explicit instantiation
    template DArray<float> preproc(DArray<float> &, float);
    template DArray<double> preproc(DArray<double> &, double);

} // namespace tomocam
