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
#include <thread>

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
        cudaHostRegister(sino.begin(), sino.bytes(), cudaHostRegisterPortable);
        cudaHostRegister(output.begin(), output.bytes(), cudaHostRegisterPortable);

        // input and output dimensions
        dim3_t idims  = sino.dims();
        dim3_t odims  = output.dims();

        // copy angles to device memory
        auto d_angles = DeviceArray_fromHost<float>(dim3_t(1, 1, idims.y), angles, 0);

        // create kernel 
        // TODO this should be somewhere else, with user input
        float beta = 12.566370614359172f;
        float radius  = 2.f;
        kernel_t kernel(radius, beta);

        // create smaller partitions
        int nStreams = 0, slcs = 0;
        MachineConfig::getInstance().update_work(idims.x, slcs, nStreams);
        std::vector<Partition<float>> sub_sinos = sino.sub_partitions(slcs);
        std::vector<Partition<float>> sub_outputs = output.sub_partitions(slcs);

        // create cudaStreams
        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }

        for (int i = 0; i < sub_sinos.size(); i++) {
            int i_stream = i % nStreams;
            stage_back_project(sub_sinos[i], sub_outputs[i], over_sample, center,
                    d_angles, kernel, streams[i_stream]);
            cudaStreamSynchronize(streams[i_stream]);
        }

        for (auto s : streams) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }

        cudaDeviceSynchronize();
        cudaHostUnregister(sino.begin());
        cudaHostUnregister(output.begin());
    }

    // inverse radon (Multi-GPU call)
    void iradon(DArray<float> &input, DArray<float> &output, float * angles,
                float center, float over_sample) {

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<float>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<float>> p2 = output.create_partitions(nDevice);

        // launch all the available devices
        std::vector<std::thread> threads;
        for (int i = 0; i < nDevice; i++) {
            threads.push_back(std::thread(iradon_, p1[i], p2[i], center, over_sample, angles, i));
        }
        
        // wait for devices to finish
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }

} // namespace tomocam
