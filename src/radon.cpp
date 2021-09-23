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
        center += ipad;

        // run batches of nStreams
        int n_parts = sub_inputs.size();
        int n_batch = ceili(n_parts, nStreams);
        for (int i = 0; i < n_batch; i++) {

            // current batch size
            int n_sub = std::min(nStreams, n_parts - i * nStreams);
            std::vector<dev_arrayc> d_volms;
            std::vector<dev_arrayc> d_sinos;

            // asynchronously copy data to device
            for (int j = 0; j < n_sub; j++) {
                auto t1 = DeviceArray_fromHostR2C(sub_inputs[i * nStreams + j], streams[j]);
                d_volms.push_back(t1);
            }

            // create device arrays for output
            for (int j = 0; j < n_sub; j++) {
                dim3_t d = sub_sinos[i * nStreams + j].dims();
                dim3_t pad_odims = dim3_t(d.x, d.y, d.z + 2 * ipad);
                auto t1 = DeviceArray_fromDims<cuComplex_t>(pad_odims, streams[j]);
                d_sinos.push_back(t1);
            }

            // asynchronously launch kernels
            for (int j = 0; j < n_sub; j++)
               stage_fwd_project(d_volms[j], d_sinos[j], ipad, center, d_angles, kernel, streams[j]);

            // asynchronously copy data from device to host
            for (int j = 0; j < n_sub; j++)
                copy_fromDeviceArrayC2R(sub_sinos[i * nStreams + j], d_sinos[j], streams[j]);

            // clean up
            for (int j = 0; j < n_sub; j++) {
                cudaStreamSynchronize(streams[j]);
                d_volms[j].free();
                d_sinos[j].free();
            }
        }

        for (auto s : streams) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
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
