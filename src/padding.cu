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
    void real_to_complex_kernel(dev_memoryF input, dev_memoryZ output) {
        int3 idx = Index3D();
        if (idx < input.dims()) {
            output[idx].x = input[idx];
        }
    }

    __global__ 
    void complex_to_real_kernel(dev_memoryZ input, dev_memoryF output) {
        int3 idx = Index3D();
        if (idx < output.dims()) {
            output[idx] = input[idx].x;
        }
    }

    // cast to complex
    dev_arrayZ real_to_cmplx(dev_arrayF &input, cudaStream_t stream) {

        // dimensions
        auto dims = input.dims();

        // allocate complex array
        dev_arrayZ output = DeviceArray_fromDims<cuComplex_t>(dims, stream);

        Grid grid(dims);
        real_to_complex_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(input, output);
        return output;
    }

    // typecast to real
    dev_arrayF cmplx_to_real(dev_arrayZ &input, cudaStream_t stream) {

        // dimensions
        auto dims = input.dims();

        // allocate real array
        dev_arrayF output = DeviceArray_fromDims<float>(dims, stream);
        
        Grid grid(dims);
        complex_to_real_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(input, output);
        return output;
    }
} // namespace tomocam
