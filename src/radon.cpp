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
#include "util.h"

namespace tomocam {

    void radon_(Partition<float> input, Partition<float> sino, float center, float over_sample,
        float *angles, int device) {

        // initalize the device
        cudaSetDevice(device);
        cudaError_t status;

        // streams and work-per-stream
        MachineConfig &cfg = MachineConfig::getInstance();
        int StreamSlices   = cfg.slicesPerStream();
        int NumStreams     = cfg.streamsPerGPU();

        // input
        dim3_t idims  = input.dims();
        float *h_data = input.begin();

        // output
        dim3_t odims  = sino.dims();
        float *f_data = sino.begin();

        // convolution kernel
        float beta = 12.566370614359172f;
        float W    = 5.f;
        kernel_t kernel = kaiser_window(W, beta, 256);

        // projection angles
        DeviceArray<float> d_angles = DeviceArray_fromHost<float>(dim3_t(1, 1, odims.y), angles, 0);

        int nStreams = 0, slcs = 0;
        if (idims.x < NumStreams) {
            slcs     = 1;
            nStreams = idims.x;
        } else if (idims.x < NumStreams * StreamSlices) {
            slcs     = idims.x / NumStreams;
            nStreams = NumStreams;
        } else {
            slcs     = StreamSlices;
            nStreams = NumStreams;
        }

        size_t istreamSize = slcs * idims.y * idims.z;
        size_t ostreamSize = slcs * odims.y * odims.z;
        dim3_t stream_idims(slcs, idims.y, idims.z);
        dim3_t stream_odims(slcs, odims.y, odims.z);

        // create cudaStreams
        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            streams.push_back(temp);
        }
        for (auto &s : streams) cudaStreamCreate(&s);

        size_t offset1 = 0;
        size_t offset2 = 0;
        int nIters     = idims.x / (nStreams * slcs);
        for (int i = 0; i < nIters; i++) {
            // launch concurrent kernels
            for (int i = 0; i < nStreams; i++) {
                offset1 = i * istreamSize;
                offset2 = i * ostreamSize;
                stage_fwd_project(h_data + offset1, f_data + offset2, stream_idims, stream_odims, over_sample, center,
                    d_angles, kernel, streams[i]);
            }
        }

        // left-over data that didn't fit into equal-sized chunks
        int nResidual = idims.x % (slcs * nStreams);
        if (nResidual > 0) {
            std::vector<int> resSlcs;

            if (nResidual < nStreams) {
                resSlcs.assign(nResidual, 1);
                nStreams = nResidual;
            } else
                resSlcs = distribute(nResidual, nStreams);

            // lauch kernels on rest of the data
            offset1 = 0;
            offset2 = 0;
            for (int i = 0; i < nStreams; i++) {
                stream_idims.x = resSlcs[i];
                stream_odims.x = resSlcs[i];
                stage_fwd_project(h_data + offset1, f_data + offset2, stream_idims, stream_odims, over_sample, center,
                    d_angles, kernel, streams[i]);
                offset1 += resSlcs[i] * idims.y * idims.z;
                offset2 += resSlcs[i] * odims.y * odims.z;
            }
            for (auto s : streams) {
                cudaStreamSynchronize(s);
                cudaStreamDestroy(s);
            }
        }
    }

    // inverse radon (Multi-GPU call)
    void radon(DArray<float> &input, DArray<float> &output, float * angles,
                float center, float over_sample) {

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<float>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<float>> p2 = output.create_partitions(nDevice);
        //
        // launch all the available devices
        std::vector<std::thread> threads;
        for (int i = 0; i < nDevice; i++) 
            threads.push_back(std::thread(radon_, p1[i], p2[i], center, over_sample, angles, i));

        // wait for devices to finish
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }

} // namespace tomocam
