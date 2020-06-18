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
#include <future>
#include <cublas_v2.h>

#include "common.h"
#include "dist_array.h"
#include "dev_array.h"
#include "machine.h"
#include "internals.h"
#include "types.h"

namespace tomocam {

    float calc_norm2(DeviceArray<float> arr,  cudaStream_t stream) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        const int incx = 1; 
        float norm = 0;
        cublasSnrm2(handle, arr.size(), arr.dev_ptr(), incx, &norm);
        cublasDestroy(handle);
        return norm;        
    }

    float norm2_(Partition<float> input, int device) {

        // initalize the device
        cudaSetDevice(device);

        // size of input and output partitions
        dim3_t idims = input.dims();

        int nStreams = 0, slcs = 0;
        MachineConfig::getInstance().update_work(idims.x, slcs, nStreams);
        std::vector<Partition<float>> sub_inputs = input.sub_partitions(slcs);

        // create cudaStreams
        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }

        int n_partitions = sub_inputs.size();
        float * norms = new float[n_partitions];
        int n_batch = n_partitions / nStreams + 1;
        for (int i = 0; i < n_batch; i++) {
            int np = std::min(nStreams, n_partitions - i * nStreams);
            
            std::vector<DeviceArray<float>> dev_arr;
            for (int j = 0; j < np; j++) 
                dev_arr.push_back(DeviceArray_fromHost(sub_inputs[i * nStreams + j], streams[j]));
                
            for (int j = 0; j < np; j++)
                norms[i * nStreams + j] = calc_norm2(dev_arr[j], streams[j]);

            for (int j = 0; j < np; j++) {
                copy_fromDeviceArray(sub_inputs[i * nStreams + j], dev_arr[j], streams[j]); 
                cudaStreamSynchronize(streams[j]);
                dev_arr[j].free();
            }
        }

        for (auto s : streams) {
            cudaStreamDestroy(s);
        }

        float p_norm = 0.f;
        for (int i = 0; i < n_partitions; i++)
            p_norm = norms[i] * norms[i];
        delete [] norms;

        return p_norm;
    }

    // inverse radon (Multi-GPU call)
    float norm2(DArray<float> &input ) {

        // pin host memory
        cudaHostRegister(input.data(), input.bytes(), cudaHostRegisterPortable);

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<float>> p1 = input.create_partitions(nDevice);

        // launch all the available devices
        std::vector<std::future<float>> futures;
        for (int i = 0; i < nDevice; i++) 
            futures.push_back(std::async(norm2_, p1[i], i));

        // wait for devices to finish
        float norm = 0;
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            norm += futures[i].get();
        }
        cudaHostUnregister(input.data());
        return std::sqrt(norm);
    }
} // namespace tomocam
