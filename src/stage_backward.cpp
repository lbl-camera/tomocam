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

#include "dist_array.h"
#include "dev_array.h"
#include "kernel.h"
#include "fft.h"
#include "internals.h"
#include "types.h"

#include "debug.cuh"

namespace tomocam {

    void stage_back_project(Partition<float> input, Partition<float> output, 
            float over_sampling, float center,
            DeviceArray<float> angles, kernel_t kernel, cudaStream_t stream) {

        // working dimensions
        dim3_t idims = input.dims();
        dim3_t odims = output.dims();
        size_t nelems = idims.x * idims.y * idims.z;
        size_t padded = (size_t)((float)idims.z * over_sampling);
        dim3_t pad_idims(idims.x, idims.y, padded);
        dim3_t pad_odims(odims.x, padded, padded);
        size_t ipad = (pad_odims.z - odims.z)/2;
        center += ipad;

    
        // input device array
        DeviceArray<cuComplex_t> d_input = DeviceArray_fromHostR2C(input, stream);

        // add zero padding
        int dim = 1;
        addPadding(d_input, ipad, dim, stream);

        // output device array
        cuComplex_t * temp;
        cudaMalloc((void **) &temp, output.size() * sizeof(cuComplex_t));
        cudaMemsetAsync(temp, 0, output.size() * sizeof(cuComplex_t), stream);    
        DeviceArray<cuComplex_t> d_output(pad_odims, temp);

        // do the acctual iverse-radon transform
        back_project(d_input, d_output, center, angles, kernel, stream);
        cudaStreamSynchronize(stream);

        // remove padding
        dim = 2;
        stripPadding(d_output, ipad, dim, stream);
        copy_fromDeviceArrayC2R(output, d_output, stream);
        cudaStreamSynchronize(stream);

        // clean up
        d_input.free();
        d_output.free();
    }
} // namespace tomocam

