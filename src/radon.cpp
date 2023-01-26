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

        // create grid for CUFINUFFT
        int ncols = odims.z;
        int nproj = odims.y;
        NUFFTGrid grid(ncols, nproj, angles, device);

        // create sub-partitions
        int nslcs = MachineConfig::getInstance().slicesPerStream();
        std::vector<Partition<float>> sub_inputs = input.sub_partitions(nslcs);
        std::vector<Partition<float>> sub_sinos = sino.sub_partitions(nslcs);
        int n_batch = sub_inputs.size();

        // create cudaStreams
        cudaStream_t istream, ostream;
        cudaStreamCreate(&istream);
        cudaStreamCreate(&ostream);

        for (int i = 0; i < n_batch; i++) {

            // copy image data to device
            auto t1 = DeviceArray_fromHost<float>(sub_inputs[i], istream);
            dev_arrayZ d_volm = real_to_cmplx(t1, istream);

            // create output array with padding
            dim3_t d = sub_sinos[i].dims();
            auto d_sino = DeviceArray_fromDims<cuComplex_t>(d, istream);

            // asynchronously launch kernels
            cudaStreamSynchronize(istream);
            project(d_volm, d_sino, center, grid); 
            cudaStreamSynchronize(ostream);
            cudaStreamSynchronize(cudaStreamPerThread);

            // cast to real data
            dev_arrayF t2 = cmplx_to_real(d_sino, ostream);

            // copy 
            copy_fromDeviceArray(sub_sinos[i], t2, ostream);
        }
        cudaStreamDestroy(istream);
        cudaStreamDestroy(ostream);
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
