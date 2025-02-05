
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
    void pad2d(Partition<T> arr, Partition<T> arr2, int npad, PadType type, int device) {

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

        #pragma omp parallel for
        for (int i = 0; i < ndevices; i++) {
            pad2d(p1[i], p2[i], npad, type, i);
        }

        // synchronize
        for (int i = 0; i < ndevices; i++) {
            SAFE_CALL(cudaSetDevice(i));
            SAFE_CALL(cudaDeviceSynchronize());
        }
        return arr2;
    }

    // explicit instantiation
    template DArray<float> pad2d(DArray<float> &, int, PadType);
    template DArray<double> pad2d(DArray<double> &, int, PadType);

} // namespace tomocam
