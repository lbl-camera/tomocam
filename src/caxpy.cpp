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

    void calc_axpy(float alpha, DeviceArray<float> x,  DeviceArray<float> y) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        const int inc = 1; 
        cublasSaxpy(handle, x.size(), &alpha, x.d_array(), inc, y.d_array(), inc);
        cublasDestroy(handle);
    }

    void axpy_(float alpha, Partition<float> x, Partition<float>, int device){
        // initalize the device
        cudaSetDevice(device);
        cudaHostRegister(x.begin(), x.bytes(), cudaHostRegisterPortable);
        cudaHostRegister(y.begin(), y.bytes(), cudaHostRegisterPortable);

        // size of input and output partitions
        dim3_t idims = x.dims();

        int nStreams = 0, slcs = 0;
        MachineConfig::getInstance().update_work(idims.x, slcs, nStreams);
        std::vector<Partition<float>> sub_xs = x.sub_partitions(slcs);
        std::vector<Partition<float>> sub_ys = y.sub_partitions(slcs);

        // create cudaStreams
        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }

        int n_partitions = sub_xs.size();
        int n_batch = n_partitions / nStreams + 1;
        for (int i = 0; i < n_batch; i++) {
            int np = std::min(nStreams, n_partitions - i * nStreams);
            
            std::vector<DeviceArray<float>> dev_x;
            std::vector<DeviceArray<float>> dev_y;
            for (int j = 0; j < np; j++) {
                dev_x.push_back(DeviceArray_fromHost(sub_xs[i * nStreams + j], streams[j]));
                dev_y.push_back(DeviceArray_fromHost(sub_ys[i * nStreams + j], streams[j]));
            }
                
            for (int j = 0; j < np; j++)
                calc_axpy(alpha, dev_x[j], dev_y[j], streams[j]);

            for (int j = 0; j < np; j++) {
                copy_fromDeviceArray(sub_ys[i * nStreams + j], dev_y[j], streams[j]); 
                cudaStreamSynchronize(streams[j]);
                dev_y[j].free();
                dev_x[j].free();
            }
        }

        for (auto s : streams) {
            cudaStreamDestroy(s);
        }
        cudaHostUnregister(x.begin());
        cudaHostUnregister(y.begin());
    }

    // y = y + alpha * x (Multi-GPU call)
    void axpy(float alpha, DArray<float> x, DArray<float> y) {

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<float>> p1 = x.create_partitions(nDevice);
        std::vector<Partition<float>> p2 = y.create_partitions(nDevice);

        // launch all the available devices
        std::vector<std::thread> threads;
        for (int i = 0; i < nDevice; i++) 
            threads.push_back(std::thread(axpy_, alpha, p1[i], p2[i], i));

        // wait for devices to finish
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }
} // namespace tomocam
