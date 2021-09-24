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
#include "fft.h"
#include "internals.h"
#include "kernel.h"
#include "types.h"

namespace tomocam {
    /* calls forward and backward projectors to calculate gradients */
    void calc_gradient(dev_arrayc &model, dev_arrayf &sino, int ipad, float center,
                         dev_arrayf &angles, kernel_t kernel, cudaStream_t stream) {

        // create device_array for forward projection
        dim3_t dims = sino.dims();

        // z-dimension should be same as padded model dimension
        dims.z = model.dims().z;
        auto proj = DeviceArray_fromDims<cuComplex_t>(dims, stream);

        // do the forward projection
        fwd_project(model, proj, center, angles, kernel, stream);

        // overwrite projection with error and redo the zero-padding
        calc_error(proj, sino, ipad, stream);

        // set d_model to zero
        cudaMemsetAsync(model.dev_ptr(), 0, model.size() * sizeof(cuComplex_t), stream);

        // backproject the error
        back_project(proj, model, center, angles, kernel, stream);

        // clean up
        proj.free();
    }
} // namespace tomocam
