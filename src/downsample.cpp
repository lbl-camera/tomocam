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
#include "internals.h"
#include "machine.h"
#include "types.h"
#include "resample.h"

namespace tomocam {

    template <typename T>
    void down_sample(Partition<T> input, Partition<T> output, int n, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // input and output dimensions
        dim3_t dims1 = input.dims();
        dim3_t dims2 = output.dims();

        // sub-partitions
        int nslcs = MachineConfig::getInstance().slicesPerStream();
        auto p1 = input.sub_partitions(nslcs);
        auto p2 = output.sub_partitions(nslcs/n);
        int n_batch = p1.size();
        
        // create cudaStreams
        cudaStream_t istream, ostream;
        cudaStreamCreate(&istream);
        cudaStreamCreate(&ostream);

        for (int i = 0; i < n_batch; i++) {
 
            // copy data to device
            auto inp = DeviceArray_fromHost<T>(p1[i], istream);
            auto out = DeviceArray_fromDims<T>(p2[i].dims(), istream);

            downsample(inp, out, n);

            // copy data back to host
            cudaStreamSynchronize(ostream);
            copy_fromDeviceArray(p2[i], out, ostream);

            // delete device_arrays
            cudaStreamSynchronize(ostream);
            inp.free();
            out.free();
        }
        cudaStreamDestroy(istream);
        cudaStreamDestroy(ostream);
    }

    // Multi-GPU calll
    template <typename T>
    DArray<T> downSample(DArray<T> &input, int n) { 

        int nDevice = MachineConfig::getInstance().num_of_gpus();

        // Allocate output array
        auto d = input.dims();
        dim3_t dims2(d.x/n, d.y/n, d.z/n);
        DArray<T> output(dims2);
        if (nDevice > output.slices()) nDevice = output.slices();

    
        auto p1 = input.create_partitions(nDevice);
        auto p2 = output.create_partitions(nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++)
            down_sample(p1[i], p2[i], n, i);


        // wait for devices to finish
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        return output;
    }

    template DArray<float> downSample(DArray<float> &, int);

} // namespace tomocam
