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
#include "kernel.h"
#include "dist_array.h"
#include "machine.h"
#include "internals.h"
#include "types.h"

namespace tomocam {

    void radon_(Partition<float> input, Partition<float> sino, float center, float over_sample,
        float *angles, int device) {

        // initalize the device
        cudaSetDevice(device);

        // size of input and output partitions
        dim3_t idims = input.dims();
        dim3_t odims = sino.dims();

        // projection angles
        auto d_angles = DeviceArray_fromHost<float>(dim3_t(1, 1, odims.y), angles, 0);

        // convolution kernel
        float beta = 12.566370614359172f; // 4π
        float radius = 2.f;
        kernel_t kernel(radius, beta);

        int nStreams = 0, slcs = 0;
        MachineConfig::getInstance().update_work(idims.x, slcs, nStreams);
        std::vector<Partition<float>> sub_inputs = input.sub_partitions(slcs);
        std::vector<Partition<float>> sub_sinos = sino.sub_partitions(slcs);

        // create cudaStreams
        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }

        // calculate padding
        int ipad = (int) ((over_sample - 1) * idims.z / 2);
        int3 pad1 = {0, ipad, ipad};
        int3 pad2 = {0, 0, ipad};
        center += ipad;

        // run batches of nStreams
        int n_parts = sub_inputs.size();
        int n_batch = ceili(n_parts, nStreams);
        for (int i = 0; i < n_batch; i++) {

            // current batch size
            int n_sub = std::min(nStreams, n_parts - i * nStreams);

            #pragma omp parallel for num_threads(n_sub)
            for (int j = 0; j < n_sub; j++) {

                // copy image data to device
                auto t1 = DeviceArray_fromHost<float>(sub_inputs[i * nStreams + j], streams[j]);
                dev_arrayc d_volm = add_paddingR2C(t1, pad1, streams[j]);

                // create output array with padding
                dim3_t d = sub_sinos[i * nStreams + j].dims();
                d.z += 2 * ipad;
                auto d_sino = DeviceArray_fromDims<cuComplex_t>(d, streams[j]);

                // asynchronously launch kernels
                fwd_project(d_volm, d_sino, center, d_angles, kernel, streams[j]);

                // remove padding from projections
                dev_arrayf t2 = remove_paddingC2R(d_sino, pad2, streams[j]);

                // copy 
                copy_fromDeviceArray(sub_sinos[i * nStreams + j], t2, streams[j]);

                // clean up
                cudaStreamSynchronize(streams[j]);
                t1.free();
                t2.free();
                d_volm.free();
                d_sino.free();
            }
        }

        for (auto s : streams)
            cudaStreamDestroy(s);
    }

    // inverse radon (Multi-GPU call)
    void radon(DArray<float> &input, DArray<float> &output, float * angles,
                float center, float over_sample) {

        // pin host memory
        cudaHostRegister(input.data(), input.bytes(), cudaHostRegisterPortable);
        cudaHostRegister(output.data(), output.bytes(), cudaHostRegisterPortable);

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        if (nDevice > input.slices()) nDevice = input.slices();

        std::vector<Partition<float>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<float>> p2 = output.create_partitions(nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) 
            radon_(p1[i], p2[i], center, over_sample, angles, i);

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
