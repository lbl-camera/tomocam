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

#include "dev_array.h"
#include "dist_array.h"
#include "internals.h"
#include "machine.h"
#include "scheduler.h"
#include "shipper.h"
#include "types.h"

namespace tomocam {

    template <typename T>
    void radon_(Partition<T> input, Partition<T> sino, NUFFT::Grid<T> &nugrid,
        int offset, int device) {

        // set device
        SAFE_CALL(cudaSetDevice(device));

        // create subpartitions
        int nparts = Machine::config.num_of_partitions(input.nslices());
        auto sub_ins = create_partitions(input, nparts);
        auto sub_outs = create_partitions(sino, nparts);

        // create data shipper to the host
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // create and scheduler
        Scheduler<Partition<T>, DeviceArray<T>> s(sub_ins);
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto [idx, d_input] = work.value();
                auto d_sino = project(d_input, nugrid, offset);

                // copy the result to the output
                // d_sino.copy_to(sub_outs[idx], copy_s);
                shipper.push(sub_outs[idx], d_sino);
            }
        }
    }

    // radon (Multi-GPU call)
    template <typename T>
    DArray<T> project(DArray<T> &input, const std::vector<T> &angles,
        int center) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > input.nslices()) nDevice = input.nslices();

        // allocate output
        int nprojs = angles.size();
        int ncols = input.ncols();
        dim3_t dims(input.nslices(), nprojs, ncols);
        DArray<T> output(dims);

        // axis offset from the center of the image
        int offset = center - ncols / 2;

        // create the nugrids
        if (offset != 0) { ncols += 2 * abs(offset); }
        std::vector<NUFFT::Grid<T>> grids(nDevice);
        for (int i = 0; i < nDevice; i++)
            grids[i] = NUFFT::Grid<T>(nprojs, ncols, angles.data(), i);

        auto p1 = create_partitions(input, nDevice);
        auto p2 = create_partitions(output, nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++)
            radon_(p1[i], p2[i], grids[i], offset, i);

        // wait for devices to finish
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        return output;
    }

    // specializations for float and double
    template DArray<float> project(DArray<float> &input,
        const std::vector<float> &angles, int center);
    template DArray<double> project(DArray<double> &input,
        const std::vector<double> &angles, int center);

} // namespace tomocam
