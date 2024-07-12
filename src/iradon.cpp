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
#include "machine.h"
#include "internals.h"
#include "types.h"

#include "scheduler.h"

namespace tomocam {

    template <typename T>
    void backproject(Partition<T> sino, Partition<T> output,
        NUFFT::Grid<T> grid, int offset, int device) {

        // select device
        cudaSetDevice(device);

        // create streams
        cudaStream_t out;
        cudaStreamCreate(&out);

        // input dimensions
        dim3_t idims = sino.dims();

        // subpartitions
        int nslcs = Machine::config.slicesPerStream();
        int nparts =
            idims.x % nslcs == 0 ? idims.x / nslcs : idims.x / nslcs + 1;
        std::vector<Partition<T>> sub_sinos = create_partitions(sino, nparts);
        std::vector<Partition<T>> sub_outputs =
            create_partitions(output, nparts);

        // start a scheduler
        Scheduler<Partition<T>, DeviceArray<T>> scheduler(sub_sinos);
        while(scheduler.has_work()) {
            auto task = scheduler.get_work();
            if (task.has_value()) {
              auto [i, d_sino] = task.value();
              auto d_recn =
                  backproject(d_sino, grid, offset, cudaStreamPerThread);
              SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
              d_recn.copy_to(sub_outputs[i], out);
            }
        }

        // wait for the thread to finish
        SAFE_CALL(cudaStreamSynchronize(out));
        SAFE_CALL(cudaStreamDestroy(out));
    }

    // back projection
    template <typename T>
    DArray<T> backproject(DArray<T> &input, const std::vector<T> &angles,
        int center) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > input.nslices()) nDevice = input.nslices();

        // output dimensions
        dim3_t dims = {input.nslices(), input.ncols(), input.ncols()};
        DArray<T> output(dims);

        // axis offset from the center of the image
        int offset = center - input.ncols() / 2;

        // create NUFFT grids
        int nprojs = angles.size();
        int ncols = input.ncols();
        if (offset != 0) { ncols += 2 * std::abs(offset); }
        std::vector<NUFFT::Grid<T>> grids(nDevice);
        for (int i = 0; i < nDevice; i++) {
            grids[i] = NUFFT::Grid<T>(nprojs, ncols, angles.data(), i);
        }

        // create partitions
        auto p1 = create_partitions(input, nDevice);
        auto p2 = create_partitions(output, nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++)
            backproject(p1[i], p2[i], grids[i], offset, i);

        // wait for devices to finish
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }

        return output;
    }

    // explicit instantiation
    template DArray<float> backproject(DArray<float> &,
        const std::vector<float> &, int);
    template DArray<double> backproject(DArray<double> &,
        const std::vector<double> &, int);

} // namespace tomocam
