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
#include "kernel.h"
#include "machine.h"
#include "types.h"

namespace tomocam {

    void gradient_(Partition<float> model, Partition<float> sino, float center,
        float over_sample, float *h_angles, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // input and output dimensions
        dim3_t idims = model.dims();
        dim3_t odims = sino.dims();

        // copy angles to the device
        auto angles = DeviceArray_fromHost<float>(dim3_t(1, 1, odims.y), h_angles, 0);

        // interpolation kernel
        float beta = 12.566370614359172f;
        float radius = 2.f;
        kernel_t kernel(radius, beta);

        int nStreams = 0, slcs = 0;
        MachineConfig::getInstance().update_work(idims.x, slcs, nStreams);
        std::vector<Partition<float>> sub_model = model.sub_partitions(slcs);
        std::vector<Partition<float>> sub_sino = sino.sub_partitions(slcs);

        // create cudaStreams
        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }

        // padding on each end
        int ipad = (int)((over_sample - 1) * idims.z / 2);
        int3 padding = {0, ipad, ipad};
        center += ipad;

        // run batches on nStreams
        int n_parts = sub_model.size();
        int n_batch = ceili(n_parts, nStreams);
        for (int i = 0; i < n_batch; i++) {

            // current batch size
            int n_sub = std::min(nStreams, n_parts - i * nStreams);

            #pragma omp parallel for num_threads(n_sub)
            for (int j = 0; j < n_sub; j++) {

                // copy model to device
                auto t1 = DeviceArray_fromHost<float>(sub_model[i * nStreams + j], streams[j]);
                dev_arrayc d_model = add_paddingR2C(t1, padding, streams[j]);

                // copy data to device
                auto d_sino = DeviceArray_fromHost<float>(sub_sino[i * nStreams + j], streams[j]);

                // calculate gradients
                calc_gradient(d_model, d_sino, ipad, center, angles, kernel, streams[j]);

                // copy data back to host
                dev_arrayf t2 = remove_paddingC2R(d_model, padding, streams[j]);
                copy_fromDeviceArray(sub_model[i * nStreams + j], t2, streams[j]);

                // delete device_arrays
                cudaStreamSynchronize(streams[j]);
                t1.free();
                t2.free();
                d_model.free();
                d_sino.free();
            }
        }

        for (auto s : streams)
            cudaStreamDestroy(s);
    }

    // Multi-GPU calll
    void gradient(DArray<float> &model, DArray<float> &sinogram, float *angles,
                  float center, float over_sample) {

        // pin host memory
        cudaHostRegister(model.data(), model.bytes(), cudaHostRegisterPortable);
        cudaHostRegister(sinogram.data(), sinogram.bytes(), cudaHostRegisterPortable);

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        if (nDevice > model.slices()) nDevice = model.slices();

        std::vector<Partition<float>> p1 = model.create_partitions(nDevice);
        std::vector<Partition<float>> p2 = sinogram.create_partitions(nDevice);

        // launch all the available devices
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            gradient_(p1[i], p2[i], center, over_sample, angles, i);
        }

        // wait for devices to finish
        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        cudaHostUnregister(model.data());
        cudaHostUnregister(sinogram.data());
    }

} // namespace tomocam
