/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley National
 * Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
 *  Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "utils.cuh"

namespace tomocam {
    __global__ 
    void real_to_complex_kernel(dev_arrayf input, dev_arrayc output, int3 padding) {
        int3 idx = Index3D();
        if (idx < input.dims()) {
            int3 idx2 = {idx.x + padding.x, idx.y + padding.y, idx.z + padding.z};
            output[idx2].x = input[idx];
        }
    }

    __global__ 
    void complex_to_real_kernel(dev_arrayc input, dev_arrayf output, int3 padding) {
        int3 idx = Index3D();
        if (idx < output.dims()) {
            int3 idx2 = {idx.x + padding.x, idx.y + padding.y, idx.z + padding.z};
            output[idx] = input[idx2].x;
        }
    }

    // cast to complex and add padding
    dev_arrayc add_paddingR2C(dev_arrayf &input, int3 padding, cudaStream_t stream) {
        // input dims
        auto dims = input.dims();

        // output dims 
        dim3_t padded = {dims.x+2*padding.x, dims.y+2*padding.y, dims.z+2*padding.z};

        // allocate padded complex array
        dev_arrayc output = DeviceArray_fromDims<cuComplex_t>(padded, stream);

        Grid grid(dims);
        real_to_complex_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(input, output, padding);
        return output;
    }

    // remove padding and typecast to real
    dev_arrayf remove_paddingC2R(dev_arrayc &input, int3 padding, cudaStream_t stream) {

        // get dimensions
        auto d = input.dims();

        // allocate updadded real array
        dim3_t dims = {d.x - 2*padding.x, d.y - 2*padding.y, d.z - 2*padding.z};
        dev_arrayf output = DeviceArray_fromDims<float>(dims, stream);
        
        Grid grid(dims);
        complex_to_real_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(input, output, padding);
        return output;
    }
} // namespace tomocam
