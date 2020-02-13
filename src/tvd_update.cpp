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

#include <thread>

#include <cuda_runtime.h>
#include <cuda.h>

#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "types.h"
#include "internals.h"

namespace tomocam {

    void total_var_(Partition<float> model, Partition<float> objfn, float p, float sigma, int device) {

        cudaError_t status;

        // initalize the device
        cudaSetDevice(device);
        cudaHostRegister(model.begin(), model.bytes(), cudaHostRegisterPortable);
        cudaHostRegister(objfn.begin(), objfn.bytes(), cudaHostRegisterPortable);
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

        // create sub-partitions of `slcs` slices each
        std::vector<Partition<float>> objs = objfn.sub_partitions(slcs);
        // create sub-partitions with halo
        std::vector<Partition<float>> mods = model.sub_partitions(slcs, 1);

        for (int i = 0; i < objs.size(); i++) {
            int i_stream = i % nStreams;
            DeviceArray<float> d_model = DeviceArray_fromHost<float>(mods[i], streams[i_stream]);
            DeviceArray<float> d_objfn = DeviceArray_fromHost<float>(objs[i], streams[i_stream]);

            add_total_var(d_model, d_objfn, p, sigma, streams[i_stream]);

            copy_fromDeviceArray<float>(objs[i], d_objfn, streams[i_stream]);
            d_model.free();
        }
            
        for (auto &s : streams) {
            cudaStreamDestroy(s);
        }
        cudaHostUnregister(model.begin());
        cudaHostUnregister(objfn.begin());
    }

    // multi-GPU call
    void add_total_var(DArray<float> &model, DArray<float> &objfn, float p, float sigma) {

        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<std::thread> threads;
        int halo = 1;
        std::vector<Partition<float>> m = model.create_partitions(nDevice, halo);
        std::vector<Partition<float>> f = objfn.create_partitions(nDevice);

        for (int i = 0; i < nDevice; i++)
            threads.push_back(std::thread(total_var_, m[i], f[i], p, sigma, i));

        for (int i = 0; i < nDevice; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            threads[i].join();
        }
    }
} // namespace tomocam
