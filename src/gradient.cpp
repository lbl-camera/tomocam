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
        float over_sample, float *angles, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // input and output dimensions
        dim3_t dims1 = model.dims();
        dim3_t dims2 = sino.dims();

        // padding on each end
        int ipad = (int)((over_sample - 1) * dims2.z / 2);
        int3 padding = {0, ipad, ipad};
        center += ipad;

        // create nufft grid
        int ncols = dims2.z + 2 * ipad;
        int nproj = dims2.y;
        NUFFTGrid grid(ncols, nproj, angles, center, device_id);    

        // sub-partitions
        int nslcs = MachineConfig::getInstance().slicesPerStream();
        auto p1 = model.sub_partitions(nslcs);
        auto p2 = sino.sub_partitions(nslcs);
        int n_batch = p1.size();
        
        // create cudaStreams
        cudaStream_t istream, ostream;
        cudaStreamCreate(&istream);
        cudaStreamCreate(&ostream);

        for (int i = 0; i < n_batch; i++) {
 
            auto t1 = DeviceArray_fromHost<float>(p1[i], istream);
            dev_arrayc d_model = add_paddingR2C(t1, padding, istream);

            // copy data to device
            auto d_sino = DeviceArray_fromHost<float>(p2[i], istream);

            // gradients are enqued in per-thread-stream
            calc_gradient(d_model, d_sino, ipad, center, grid);
            cudaStreamSynchronize(cudaStreamPerThread);

            // copy data back to host
            cudaStreamSynchronize(ostream);
            dev_arrayf t2 = remove_paddingC2R(d_model, padding, ostream);
            copy_fromDeviceArray(p1[i], t2, ostream);

            // delete device_arrays
            cudaStreamSynchronize(ostream);
            t1.free();
            t2.free();
            d_model.free();
            d_sino.free();
        }
        cudaStreamDestroy(istream);
        cudaStreamDestroy(ostream);
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
