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
#include "types.h"
#include "toeplitz.h"

namespace tomocam {

    template <typename T>
    void gradient2_(Partition<T> f, Partition<T> sinoT, Partition<T> df,
        const PointSpreadFunction<T> &psf, int device_id) {

        // set device
        SAFE_CALL(cudaSetDevice(device_id));

        // sub-partitions
        int nparts = Machine::config.num_of_partitions(sinoT.nslices());
        auto p1 = create_partitions(f, nparts);
        auto p2 = create_partitions(sinoT, nparts);
        auto p3 = create_partitions(df, nparts);

        // create cuda streams
        cudaStream_t work_s = cudaStreamPerThread;
        cudaStream_t copy_s;
        SAFE_CALL(cudaStreamCreate(&copy_s));

        // creater a scheduler, and assign work
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> s(p1, p2);
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto [idx, d_f, d_sinoT] = work.value();

                // compute gradient
                auto tmp = psf.convolve(d_f, work_s);
                auto d_g = tmp.subtract(d_sinoT, work_s);

                // synchronize work stream
                SAFE_CALL(cudaStreamSynchronize(work_s));

                // copy gradient to host
                d_g.copy_to(p3[idx], copy_s);
            }
        }

        // synchronize and destroy copy stream
        SAFE_CALL(cudaStreamSynchronize(copy_s));
        SAFE_CALL(cudaStreamDestroy(copy_s));
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
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            gradient2_<T>(p1[i], p2[i], p3[i], psfs[i], i);
        }

        // wait for devices to finish
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
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
