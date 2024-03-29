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

#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>

#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "types.h"
#include "internals.h"

namespace tomocam {

    void tv_func_(Partition<float> model, Partition<float> objfn, 
        float p, float sigma, float lam, int device) {

        cudaError_t status;

        // initalize the device
        cudaSetDevice(device);

        //  output
        dim3_t idims  = objfn.dims();

        int nStreams = 0;
        int slcs     = 0;
        MachineConfig::getInstance().update_work(idims.x, slcs, nStreams);

        //  stream size
        std::vector<cudaStream_t> streams;
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }

        // create sub-partitions with halo
        std::vector<Partition<float>> mods = model.sub_partitions(slcs, 1);
     
        // create sub-partitions of `slcs` slices each
        std::vector<Partition<float>> objs = objfn.sub_partitions(slcs);

        // run batches of nStreams
        int n_parts = objs.size();
        int n_batch = ceili(n_parts, nStreams);
        for (int i = 0; i < n_batch; i++) {
          
            // current batch size
            int n_sub = std::min(nStreams, n_parts - i * nStreams);
            std::vector<dev_arrayf> d_model;
            std::vector<dev_arrayf> d_objfn;

            // copy model to device array
            for (int j = 0; j < n_sub; j++) {
                auto t1 = DeviceArray_fromHost<float>(mods[i * nStreams + j], streams[j]);
                d_model.push_back(t1);
            }

            // copy objective function to devie array
            for (int j = 0; j < n_sub; j++) {
                auto t1 = DeviceArray_fromHost<float>(objs[i * nStreams + j], streams[j]);
                d_objfn.push_back(t1);
            }

            // calcuate constraints and update the objective function in-place
            for (int j = 0; j < n_sub; j++)
                add_tv_func(d_model[j], d_objfn[j], p, sigma, lam, streams[j]);

            // copy data back to host memeory
            for (int j = 0; j < n_sub; j++) {
                cudaStreamSynchronize(streams[j]);
                copy_fromDeviceArray<float>(objs[i * nStreams + j], d_objfn[j], streams[j]);
                d_model[j].free();
                d_objfn[j].free();
            }
        }
            
        for (auto &s : streams) {
            cudaStreamDestroy(s);
        }
    }

    // multi-GPU call
    void add_tv_func(DArray<float> &model, DArray<float> &objfn,
            float p, float sigma, float lam) {

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        if (nDevice > model.slices()) nDevice = model.slices();
        int halo = 1;

        cudaHostRegister(model.data(), model.bytes(), cudaHostRegisterPortable);
        cudaHostRegister(objfn.data(), objfn.bytes(), cudaHostRegisterPortable);
        std::vector<Partition<float>> m = model.create_partitions(nDevice, halo);
        std::vector<Partition<float>> f = objfn.create_partitions(nDevice);

        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++){
            cudaSetDevice(i);
            tv_func_(m[i], f[i], p, sigma, lam, i);
        }

        #pragma omp parallel for num_threads(nDevice)
        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        cudaHostUnregister(model.data());
        cudaHostUnregister(objfn.data());
    }
} // namespace tomocam
