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
#include "fft.h"
#include "internals.h"
#include "types.h"
#include "nufft.h"

#include "debug.cuh"

namespace tomocam {

    void back_project(dev_arrayc &sino, dev_arrayc &image, float center, 
                NUFFTGrid &grid) {

        // dims
        dim3_t dims = image.dims();
    
        // fftshift
        fftshift1D(sino, cudaStreamPerThread);

        // 1-D fft
        auto cufft_plan = fftPlan1D(sino.dims());
        SAFE_CUFFT_CALL(cufftExecC2C(cufft_plan, sino.dev_ptr(), sino.dev_ptr(), CUFFT_FORWARD));
        SAFE_CUFFT_CALL(cufftDestroy(cufft_plan));

        // center shift
        ifftshift_center(sino, center, cudaStreamPerThread);

        // normalize after FFT
        rescale(sino, 1./static_cast<float>(dims.z), cudaStreamPerThread);

        // nufft type 1
        cufinufftf_plan cufinufft_plan;
        NUFFT_CALL(nufftPlan1(dims, grid, cufinufft_plan));
        NUFFT_CALL(cufinufftf_execute(sino.dev_ptr(), image.dev_ptr(), cufinufft_plan));
        NUFFT_CALL(cufinufftf_destroy(cufinufft_plan));
    }
} // namespace tomocam
