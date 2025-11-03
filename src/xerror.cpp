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


#include <future>
#include <vector>

#include <cuda_runtime.h>

#include "dev_array.h"
#include "dist_array.h"
#include "partition.h"
#include "machine.h"
#include "scheduler.h"

#include "timer.h"

namespace tomocam {

    template <typename T>
    T xerror_(Partition<T> curr, Partition<T> prev, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // sub-partitions
        int nslcs = Machine::config.num_of_partitions(curr.dims(), curr.bytes());
        auto p1 = create_partitions(curr, nslcs);
        auto p2 = create_partitions(prev, nslcs);
        T sum = 0;

        // create a scheduler
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> scheduler(p1, p2);

        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto[idx, d_curr, d_prev] = work.value();
                auto diff = d_curr - d_prev;
                sum += diff.norm2();
            }
        }
        return sum;
    }

    // Multi-GPU calll
    template <typename T>
    T xerror(DArray<T> &xcurr, DArray<T> &xprev) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > xcurr.nslices()) nDevice = xcurr.nslices();

        // create one partition per device
        auto p1 = create_partitions(xcurr, nDevice);
        auto p2 = create_partitions(xprev, nDevice);

        std::vector<std::future<T>> results(nDevice);
        for (int i = 0; i < nDevice; i++) {
            results[i] = std::async(std::launch::async, xerror_<T>, p1[i], p2[i], i);
        }

        Machine::config.barrier();
        T xerr = 0;
        for (int i = 0; i < nDevice; i++) {
            xerr += results[i].get();
        }
        return xerr;
    }

    // Explicit instantiation
    template float xerror(DArray<float> &, DArray<float> &);
    template double xerror(DArray<double> &, DArray<double> &);
} // namespace tomocam
