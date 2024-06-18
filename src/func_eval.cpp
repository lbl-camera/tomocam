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
#include <future>

#include "dev_array.h"
#include "dist_array.h"
#include "internals.h"
#include "machine.h"
#include "types.h"
#include "toeplitz.h"

namespace tomocam {

    float func_value(Partition<float> recon, Partition<float> sinoT, const SpreadFunc psf, int device_id) {

        // set device
        cudaSetDevice(device_id);

        // input and output dimensions
        dim3_t dims1 = recon.dims();
        dim3_t dims2 = sinoT.dims();

        // sub-partitions
        int nslcs = Machine::config.slicesPerStream();
        auto p1 = recon.sub_partitions(nslcs);
        auto p2 = sinoT.sub_partitions(nslcs);
        int n_batch = p1.size();
        
        float sum = 0;
        for (int i = 0; i < n_batch; i++) {
            // create cudaStreams
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            // copy data to device
            auto d_recon = DeviceArray_fromHost<float>(p1[i], stream);
            auto d_sinoT = DeviceArray_fromHost<float>(p2[i], stream);

            // function evalulation
            auto temp = psf.convolve(d_recon, stream);
            sum += (d_recon.dot(temp, stream) -  2 * d_recon.dot(d_sinoT, stream));

            // delete stream
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        return sum;
    }


    // Multi-GPU calll
    float FunctionValue(DArray<float> &recon, DArray<float> &sinoT, 
                  const SpreadFunc *psf, float sino_sq) { 

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > recon.nslices()) nDevice = recon.nslices();

        std::vector<Partition<float>> p1 = recon.create_partitions(nDevice);
        std::vector<Partition<float>> p2 = sinoT.create_partitions(nDevice);

        // std::async 
        std::vector<std::future<float>> retval;
        for (int i = 0; i < nDevice; i++)
            retval.emplace_back(std::async(func_value, p1[i], p2[i], psf[i], i));

        // wait for devices to finish
        float fval = 0;
        for (int i = 0; i < nDevice; i++) {
            retval[i].wait();
            fval += retval[i].get();
        }
        return (fval + sino_sq);
    }

} // namespace tomocam
