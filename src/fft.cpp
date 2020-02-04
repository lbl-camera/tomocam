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

#include <cuda_runtime.h>
#include <cufft.h>
#include <thread>

#include "fft.h"
#include "machine.h"
#include "util.h"

namespace tomocam {
    cufftHandle fftPlan1D(dim3_t dims) {
        // order: nslc, ncol, nrow
        int rank    = 1;
        int n[]     = {dims.z};
        int istride = 1;
        int ostride = 1;
        int idist   = dims.z;
        int odist   = dims.z;
        int batches = dims.x * dims.y;

        cufftHandle plan;
        cufftResult status = cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, batches);
        if (status != CUFFT_SUCCESS) {
            std::cerr << "Failed to make a plan. Error code: " << status << std::endl;
            throw status;
        }
        return plan;
    }

    cufftHandle fftPlan2D(dim3_t dims) {
        // order: nslc, ncol, nrow
        int rank    = 2;
        int n[]     = {dims.y, dims.z};
        int istride = 1;
        int ostride = 1;
        int idist   = dims.y * dims.z;
        int odist   = dims.y * dims.z;
        int batches = dims.x;

        cufftHandle plan;
        cufftResult status = cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, batches);
        if (status != CUFFT_SUCCESS) {
            std::cerr << "Failed to make a plan. Error code: " << status << std::endl;
            throw status;
        }
        return plan;
    }

    void DArrayFFT(Partition<complex_t> input, Partition<complex_t> output, int FFT_DIRECTION, bool is_1D, int device) {

        // set device on multi-gpu machine
        cudaError_t error;
        cufftResult status;
        cudaSetDevice(device);

        int StreamSlices = MachineConfig::getInstance().slicesPerStream();
        int NumStreams   = MachineConfig::getInstance().streamsPerGPU();

        std::vector<cufftHandle> plans;
        std::vector<cudaStream_t> streams;

        dim3_t pDims = input.dims();

        int slcs     = 0;
        int nStreams = 0;
        MachineConfig::getInstance().update_work(pDims.x, slcs, nStreams);

        dim3_t dims(slcs, pDims.y, pDims.z);
        size_t streamSize  = slcs * dims.y * dims.z;
        size_t streamBytes = streamSize * sizeof(cufftComplex);

        // create cuda streams
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }

        cufftHandle p;
        for (int i = 0; i < nStreams; i++) {
            if (is_1D) p = fftPlan1D(dims);
            else
                p = fftPlan2D(dims);
            cufftSetStream(p, streams[i]);
            plans.push_back(p);
        }

        // host data from partiion
        complex_t *h_data = input.begin();
        complex_t *f_data = output.begin();

        // device data buffer
        cufftComplex *d_data = NULL;
        cudaMalloc((void **)&d_data, nStreams * streamBytes);

        int offset = 0;
        int nIters = pDims.x / (nStreams * slcs);

        for (int it = 0; it < nIters; it++) {
            for (int i = 0; i < nStreams; ++i) {
                offset = i * streamSize;
                error  = cudaMemcpyAsync(d_data + offset, h_data + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);
                if (error != cudaSuccess) {
                    std::cerr << "Failed to copy memory to device " << error << std::endl;
                    throw error;
                }
            }

            // FFT of a signal
            for (int i = 0; i < nStreams; ++i) {
                offset = i * streamSize;
                status = cufftExecC2C(plans[i], d_data + offset, d_data + offset, FFT_DIRECTION);
                if (status != CUFFT_SUCCESS) {
                    std::cerr << "Failed to execute FFT. Error code " << status << std::endl;
                    throw status;
                }
            }
            // copy data back to host
            for (int i = 0; i < nStreams; i++) {
                offset = i * streamSize;
                error  = cudaMemcpyAsync(f_data + offset, d_data + offset, streamBytes, cudaMemcpyDeviceToHost, streams[i]);
                if (error != cudaSuccess) {
                    std::cerr << "Failed to copy memory from device " << error << std::endl;
                    throw error;
                }
            }
            h_data += nStreams * streamSize;
            f_data += nStreams * streamSize;
        }
        // destroy plans, since they are irrelevent now
        for (auto &p : plans) cufftDestroy(p);

        // transform residual slices that didn't fit nicely
        int nResidual = pDims.x % (slcs * nStreams);
        if (nResidual > 0) {

            // make new plans, since we discared old plans
            std::vector<int> resSlcs;
            std::vector<cufftHandle> planB;

            if (nResidual < nStreams) {
                resSlcs.assign(nResidual, 1);
                nStreams = nResidual;
            } else
                resSlcs = distribute(nResidual, nStreams);

            // create plans with unequal batches
            for (int i = 0; i < nStreams; i++) {
                dims.x = resSlcs[i];
                if (is_1D) p = fftPlan1D(dims);
                else
                    p = fftPlan2D(dims);
                cufftSetStream(p, streams[i]);
                planB.push_back(p);
            }

            // move data to devices
            offset = 0;
            for (int i = 0; i < nStreams; i++) {
                streamSize = resSlcs[i] * dims.y * dims.z;
                error      = cudaMemcpyAsync(d_data + offset, h_data + offset, streamSize * sizeof(cufftComplex),
                    cudaMemcpyHostToDevice, streams[i]);
                if (error != cudaSuccess) {
                    std::cerr << "Failed to copy memory to device " << error << std::endl;
                    throw error;
                }
                offset += streamSize;
            }

            // Run fftexec
            offset = 0;
            for (int i = 0; i < nStreams; i++) {
                status = cufftExecC2C(planB[i], d_data + offset, d_data + offset, FFT_DIRECTION);
                if (status != CUFFT_SUCCESS) {
                    std::cerr << "Failed to execute FFT. " << status << std::endl;
                    throw status;
                }
                offset += resSlcs[i] * dims.y * dims.z;
            }

            // copy data back to host
            offset = 0;
            for (int i = 0; i < nStreams; i++) {
                streamSize = resSlcs[i] * dims.y * dims.z;
                error      = cudaMemcpyAsync(f_data + offset, d_data + offset, streamSize * sizeof(cufftComplex),
                    cudaMemcpyDeviceToHost, streams[i]);
                if (error != cudaSuccess) {
                    std::cerr << "Failed to copy memory from device " << error << std::endl;
                    throw error;
                }
                offset += streamSize;
            }
            for (auto &p : planB) cufftDestroy(p);
        }

        for (auto &s : streams) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
        cudaFree(d_data);
    }

    void fft1d(DArray<complex_t> &input, DArray<complex_t> &output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();

        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        std::vector<std::thread> threads;

        for (int i = 0; i < p1.size(); i++) {
            int device = i % nDevice;
            threads.push_back(std::thread(DArrayFFT, p1[i], p2[i], CUFFT_FORWARD, true, device));
        }

        // wait for devices to finish
        for (int i = 0; i < p1.size(); i++) {
            int device = i % nDevice;
            cudaSetDevice(device);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }

    void ifft1d(DArray<complex_t> &input, DArray<complex_t> &output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();

        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        std::vector<std::thread> threads;

        for (int i = 0; i < p1.size(); i++) {
            unsigned device = i % nDevice;
            threads.push_back(std::thread(DArrayFFT, p1[i], p2[i], CUFFT_INVERSE, true, device));
        }

        // wait for devices to finish
        for (int i = 0; i < p1.size(); i++) {
            int device = i % nDevice;
            cudaSetDevice(device);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }

    void fft2d(DArray<complex_t> &input, DArray<complex_t> &output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();

        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        std::vector<std::thread> threads;

        for (int i = 0; i < p1.size(); i++) {
            unsigned device = i % nDevice;
            threads.push_back(std::thread(DArrayFFT, p1[i], p2[i], CUFFT_FORWARD, false, device));
        }

        // wait for devices to finish
        for (int i = 0; i < p1.size(); i++) {
            int device = i % nDevice;
            cudaSetDevice(device);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }

    void ifft2d(DArray<complex_t> &input, DArray<complex_t> &output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();

        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        std::vector<std::thread> threads;

        for (int i = 0; i < p1.size(); i++) {
            unsigned device = i % nDevice;
            threads.push_back(std::thread(DArrayFFT, p1[i], p2[i], CUFFT_INVERSE, false, device));
        }

        // wait for devices to finish
        for (int i = 0; i < p1.size(); i++) {
            int device = i % nDevice;
            cudaSetDevice(device);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }

} // namespace tomocam
