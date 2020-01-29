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

#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "types.h"

namespace tomocam {

    void update_total_var(Partition<float> input, Partition<float> output, float p, float sigma, int device) {

        // initalize the device
        cudaSetDevice(device);
        cudaError_t status;

        // streams and work-per-stream
        int StreamSlices = MachineConfig::getInstance().slicesPerStream();
        int NumStreams   = MachineConfig::getInstance().streamsPerGPU();

        // input
        dim3_t idims  = input.dims();
        float *h_data = input.begin();

        //  output
        dim3_t odims  = output.dims();
        float *f_data = output.begin();

        int nStreams = 0;
        int slcs     = 0;
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

        //  stream size
        size_t streamSize = slcs * idims.y * idims.z;

        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            streams.push_back(temp);
        }
        for (auto s & : streams) cudaCreateStream(&s);

        // buffers for input and output
        complex_t *d_input  = NULL;
        complex_t *d_output = NULL;
        status              = cudaMalloc((void **)d_input, nStreams * streamSize * sizeof(complex_t));
        status              = cudaMalloc((void **)d_output, nStreams * streamSize * sizeof(complex_t));

        size_t offset    = 0;
        int nIters       = idims.x / (nStreams * slcs);
        int3 stream_dims = make_int3(slcs, idims.y, idims.z);
        for (int iter = 0; iter < nIters; iter++) {
            // copy data to streams
            for (int i = 0; i < nStreams; i++) {
                offset = i * streamSize;
                status = cudaMemcpyAsync(
                    d_input + offset, h_data + offset, streamSize * sizeof(complex_t), cudaMemcpyHostToDevice, streams[i]);
                if (status != cudaSuccess) {
                    std::cerr << "Error! failed to copy data to device. " << status << std::endl;
                    throw status;
                }
            }

            // launch concurrent kernels
            for (int i = 0; i < nStreams; i++) {
                offset = i * streamSize;
                addTVD(stream_dims, p, sigma, input + offset, output + offset, streams[i]);
            }

            // copy data back to host
            for (int i = 0; i < nStreams; i++) {
                offset = i * streamSize;
                status = cudaMemcpyAsync(
                    f_data + offset, d_output + offset, streamSize * sizeof(complex_t), cudaMemcpyDeviceToHost, streams[i]);
                if (status != cudaSuccess) {
                    std::cerr << "Error! failed to copy data from device. " << status << std::endl;
                    throw status;
                }
            }
            h_data += nStreams * streamSize;
            f_data += nStreams * streamSize;
        }

        int nResidual = idims.x % (slcs * nStreams);
        if (nResidual > 0) {
            std::vector<int> resSlcs;

            if (nResidual < nStreams) {
                resSlcs.assign(nResidual, 1);
                nStreams = nResidual;
            } else
                resSlcs = distribute(nResidual, nStreams);

            offset = 0;
            for (int i = 0; i < nStreams; i++) {
                streamSize = resSlcs[i] * idims.y * idims.z;
                status     = cudaMemcpyAsync(
                    d_data + offset, h_data + offset, streamSize * sizeof(complex_t), cudaMemcpyHostToDevice, streams[i]);
                if (status != cudaSuccess) { throw status; }
                offset += streamSize;
            }

            // lauch kernels on rest of the data
            offset = 0;
            for (int i = 0; i < nStreams; i++) {
                streamSize    = resSlcs[i] * idims.y * idims.z;
                stream_dims.x = resSlcs[i];
                addTVD(stream_dims, p, sigma, input + offset, output + offset, streams[i]);
                offset += streamSize;
            }

            offset = 0;
            for (int i = 0; i < nStreams; i++) {
                streamSize = resSlcs[i] * odims.y * odims.z;
                status     = cudaMemcpyAsync(
                    f_data + offset, d_output + offset, streamSize * sizeof(complex_t), cudaMemcpyDeviceToHost, streams[i]);
                if (status != cudaSuccess) { throw status; }
                offset += streamSize;
            }
        }
        cudaFree(d_input);
        cudaFree(d_output);
        for (auto &s : streams) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
    }
} // namespace tomocam
