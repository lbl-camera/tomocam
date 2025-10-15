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
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 *Software to reproduce, distribute copies to the public, prepare derivative
 *works, and perform publicly and display publicly, and to permit other to do
 *so.
 *---------------------------------------------------------------------------------
 */

#include <thread>
#include <vector>

#include "dev_array.h"
#include "dist_array.h"
#include "fft.h"
#include "gpu/padding.cuh"
#include "gpu/resample.cuh"
#include "machine.h"
#include "multiproc.h"
#include "partition.h"
#include "scheduler.h"
#include "shipper.h"

namespace tomocam {

    template <typename T>
    DArray<T> downsample(DArray<T> &sino, int skip) {

        if (skip == 0) return sino;

        // size of downsampled array
        int nslices = std::max(sino.nslices() / skip, 1);
        int nrows = sino.nrows();
        int ncols = sino.ncols() / skip;
        // ensure last dimension is odd
        if (ncols % 2 == 0) ncols--;

        DArray<T> downsampled({nslices, nrows, ncols});

#pragma omp parallel for
        for (int i = 0; i < nslices; i++) {
            for (int j = 0; j < nrows; j++) {
                for (int k = 0; k < ncols; k++) {
                    downsampled(i, j, k) = sino(skip * i, j, skip * k);
                }
            }
        }
        return downsampled;
    }

    // explicit instantiation
    template DArray<float> downsample(DArray<float> &, int);
    template DArray<double> downsample(DArray<double> &, int);

    template <typename T>
    void fftdownsamp_(Partition<T> sino, Partition<T> downsampled, int factor,
        int device) {

        // set the device
        SAFE_CALL(cudaSetDevice(device));

        // manually create Partitions of 1-slice each
        dim3_t dims1 = {1, sino.nrows(), sino.ncols()};
        dim3_t dims2 = {1, downsampled.nrows(), downsampled.ncols()};
        std::vector<Partition<T>> p1;
        std::vector<Partition<T>> p2;
        for (auto i = 0; i < downsampled.nslices(); i++) {
            p1.push_back(Partition(dims1, sino.slice(i * factor)));
            p2.push_back(Partition<T>(dims2, downsampled.slice(i)));
        }
        assert(p1.size() == downsampled.nslices());
        auto crop = (sino.ncols() - downsampled.ncols()) / 2;

        // create shhipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // create scheduler
        Scheduler<Partition<T>, DeviceArray<T>> s(p1);
        while (s.has_work()) {
            auto work = s.get_work();
            if (work.has_value()) {
                auto [idx, d_sino] = work.value();

                // perform the fft
                auto d_sino_ft = rfft1D(d_sino);

                // crop high frequencies
                d_sino_ft = gpu::unpad1d(d_sino_ft, crop, PadType::RIGHT);

                // inverse fft
                auto d_down = irfft1D<T>(d_sino_ft, downsampled.ncols());

                // push the result to the shipper
                shipper.push(p2[idx], d_down);
            }
        }
    }

    template <typename T>
    DArray<T> fftdownsamp(DArray<T> &sino, int factor) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > sino.nslices()) nDevice = sino.nslices();

        int nslices = std::max(sino.nslices() / factor, 1);
        int ncols = sino.ncols() / factor;
        if (ncols % 2 == 0) { ncols -= 1; }
        //
        auto dims = dim3_t(nslices, sino.nrows(), ncols);
        DArray<T> downsampled(dims);

        auto p1 = create_partitions(sino, nDevice);
        auto p2 = create_partitions(downsampled, nDevice);

        std::vector<std::thread> threads(nDevice);
        for (int i = 0; i < nDevice; i++)
            threads[i] = std::thread(fftdownsamp_<T>, p1[i], p2[i], factor, i);

        // wait for all the threads to finish
        for (auto &t : threads) { t.join(); }

        return downsampled;
    }

    // explicit instantiation
    template DArray<float> fftdownsamp(DArray<float> &, int);
    template DArray<double> fftdownsamp(DArray<double> &, int);

    template <typename T>
    void upsample(Partition<T> sol, Partition<T> upsampled, int device) {

        SAFE_CALL(cudaSetDevice(device));

        int nparts = Machine::config.num_of_partitions(upsampled.nslices());
        auto p1 = create_partitions(sol, nparts, 2);
        auto p2 = create_partitions(upsampled, nparts);

        // create shhipper
        GPUToHost<Partition<T>, DeviceArray<T>> shipper;

        // create scheduler
        Scheduler<Partition<T>, DeviceArray<T>> s(p1);
        while (s.has_work()) {

            auto work = s.get_work();
            if (work.has_value()) {
                auto [idx, d_s] = work.value();
                auto d_up = gpu::lanczos_upsampling(d_s, p2[idx].dims());
                shipper.push(p2[idx], d_up);
            }
        }
    }

    template <typename T>
    DArray<T> upsample(DArray<T> &sol, dim3_t dims) {

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > sol.nslices()) nDevice = sol.nslices();

        auto dims2 = sol.dims();
#ifdef MULTIPROC
        int myrank = multiproc::mp.myrank();
        int size = multiproc::mp.nprocs();
        if (myrank > 0) dims2.x += 2;
        if (myrank < size - 1) dims2.x += 2;
        int start = myrank == 0 ? 0 : 2;

        DArray<T> sol2(dims2);
        std::copy(sol.begin(), sol.end(), sol2.slice(start));
        sol2.update_neigh_proc();
        auto p1 = create_partitions(sol2, nDevice, 2);
#else
        auto p1 = create_partitions(sol, nDevice, 2);
#endif
        DArray<T> upsampled(dims);
        auto p2 = create_partitions(upsampled, nDevice);

#pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            upsample(p1[i], p2[i], i);
        }

// wait for all the devies to finish
#pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
        }

        return upsampled;
    }
    // explicit instantiation
    template DArray<float> upsample(DArray<float> &, dim3_t);
    template DArray<double> upsample(DArray<double> &, dim3_t);

} // namespace tomocam
