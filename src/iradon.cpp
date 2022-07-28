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
#include "kernel.h"
#include "dist_array.h"
#include "machine.h"
#include "internals.h"
#include "types.h"

namespace tomocam {

    void iradon_(Partition<float> sino, Partition<float> output, float center, float over_sample,
        float *angles, int device) {

        // select device
        cudaSetDevice(device);

        // input and output dimensions
        dim3_t idims  = sino.dims();
        dim3_t odims  = output.dims();
        
        // padding for oversampling
        int ipad = (int) ((over_sample-1) * idims.z / 2);
        int3 pad1 = {0, 0, ipad};
        int3 pad2 = {0, ipad, ipad};
        center += ipad;

        // subpartitions
        int nslcs = MachineConfig::getInstance().slicesPerStream();
        std::vector<Partition<float>> sub_sinos = sino.sub_partitions(nslcs);
        std::vector<Partition<float>> sub_outputs = output.sub_partitions(nslcs);
        int n_batch = sub_sinos.size();

        // nufft grid
        int ncols = idims.z + 2 * ipad;
        int nproj = idims.y;
        NUFFTGrid grid(ncols, nproj, angles, center, device);

        // create cudaStreams
        cudaStream_t istream, ostream;
        cudaStreamCreate(&istream);
        cudaStreamCreate(&ostream);

        for (int i = 0; i < n_batch; i++) {

            // asynchronously copy data to device
            auto t1 = DeviceArray_fromHost(sub_sinos[i], istream);
            dev_arrayc d_sino = add_paddingR2C(t1, pad1, istream);

            // allocate output array on device
            dim3_t d = sub_outputs[i].dims();
            dim3_t pad_odims = dim3_t(d.x, d.y + 2 * ipad, d.z + 2 * ipad);
            auto d_recn = DeviceArray_fromDims<cuComplex_t>(pad_odims, istream);

            // asynchronously launch kernels
            cudaStreamSynchronize(istream);
            back_project(d_sino, d_recn, center, grid);
            cudaStreamSynchronize(cudaStreamPerThread);

            // remove padding
            dev_arrayf t2 = remove_paddingC2R(d_recn, pad2, ostream);

            // asynchronously copy data back to host
            copy_fromDeviceArray(sub_outputs[i], t2, ostream);
            cudaStreamSynchronize(ostream);
           
            t1.free();
            t2.free();
            d_sino.free();
            d_recn.free();
        }

        cudaStreamDestroy(istream);
        cudaStreamDestroy(ostream);
        cudaDeviceSynchronize();
    }

    // inverse radon (Multi-GPU call)
    void iradon(DArray<float> &input, DArray<float> &output, float * angles,
                float center, float over_sample) {

        // pin host memeory
        cudaHostRegister(input.data(), input.bytes(), cudaHostRegisterPortable);
        cudaHostRegister(output.data(), output.bytes(), cudaHostRegisterPortable);

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        if (nDevice > input.slices()) nDevice = input.slices();

        std::vector<Partition<float>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<float>> p2 = output.create_partitions(nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            iradon_(p1[i], p2[i], center, over_sample, angles, i);
        }
        
        // wait for devices to finish
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        cudaHostUnregister(input.data());
        cudaHostUnregister(output.data());
    }

} // namespace tomocam
