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

    void up_sample(Partition<float> input, Partition<float> output, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // input and output dimensions
        dim3_t dims1 = input.dims();
        dim3_t dims2 = output.dims();

        // sub-partitions
        int nslcs = MachineConfig::getInstance().slicesPerStream();
        auto p1 = input.sub_partitions(nslcs);
        auto p2 = output.sub_partitions(2*nslcs);
        int n_batch = p1.size();
        
        // create cudaStreams
        cudaStream_t istream, ostream;
        cudaStreamCreate(&istream);
        cudaStreamCreate(&ostream);

        for (int i = 0; i < n_batch; i++) {
 
            // copy data to device
            auto inp = DeviceArray_fromHost<float>(p1[i], istream);
            auto out = DeviceArray_fromDims<float>(p2[i].dims(), cudaStreamPerThread);

            upsample(inp, out);

            // copy data back to host
            cudaStreamSynchronize(ostream);
            copy_fromDeviceArray(p2[i], out, ostream);

            // delete device_arrays
            cudaStreamSynchronize(ostream);
        }
        cudaStreamDestroy(istream);
        cudaStreamDestroy(ostream);
    }

    // Multi-GPU calll
    DArray<float> upSample(DArray<float> &input) { 

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        if (nDevice > input.slices()) nDevice = input.slices();

        // Allocate output array
        auto d = input.dims();
        dim3_t dims2(2*d.x, 2*d.y, 2*d.z);
        DArray<float> output(dims2);

    
        auto p1 = input.create_partitions(nDevice);
        auto p2 = output.create_partitions(nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            up_sample(p1[i], p2[i], i);
        }

        // wait for devices to finish
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        return output;
    }

} // namespace tomocam
