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

    /* zero pads arrays and remves padding after calculations */ 
    void stage_back_project(dev_arrayc sinogram, dev_arrayc &volume, int ipad, float center,
            dev_arrayf angles, kernel_t kernel, cudaStream_t stream) {

        // add zero padding
        addPadding(sinogram, ipad, 1, stream);

        // do the actual iverse-radon transform
        back_project(sinogram, volume, center, angles, kernel, stream);

        // remove padding
        stripPadding(volume, ipad, 2, stream);
    }


    /* zero pads arrays and remves padding after calculations */ 
    void stage_fwd_project(dev_arrayc volume, dev_arrayc &sinos, int ipad,
        float center, dev_arrayf angles, kernel_t kernel, cudaStream_t stream) {

        // pad input array with zeros
        addPadding(volume, ipad, 2, stream);

        // do the actual forward projection
        fwd_project(volume, sinos, center, angles, kernel, stream);

        // remove padding
        stripPadding(sinos, ipad, 1, stream); 

    }

    /* calls forward and backward projectors to calculate gradients */
    void calc_gradient(dev_arrayc &model, dev_arrayf data, int ipad, float center,
                         dev_arrayf angles, kernel_t kernel, cudaStream_t stream) {

        // zero pad model
        addPadding(model, ipad, 2, stream);

        // create device_array for singrams
        dim3_t sino_dims = data.dims();

        // z-dimension should be same as padded model dimension
        sino_dims.z = model.dims().z;
        auto sino = DeviceArray_fromDims<cuComplex_t>(sino_dims, stream);

        // do the actual forward projection
        fwd_project(model, sino, center, angles, kernel, stream);

        // overwrite d_sino with error and redo the zero-padding
        calc_error(sino, data, ipad, stream);

        // set d_model to zero
        cudaMemsetAsync(model.dev_ptr(), 0, model.size() * sizeof(cuComplex_t), stream);

        // backproject the error
        back_project(sino, model, center, angles, kernel, stream);

        // remove padding
        stripPadding(model, ipad, 2, stream);

        // clean up
        sino.free();
    }
} // namespace tomocam
