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

#ifndef TOMOCAM_FFT__H
#define TOMOCAM_FFT__H

#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

#include "dist_array.h"
#include "dev_array.h"
#include "types.h"

namespace tomocam {

    inline cufftHandle fftPlan1D(dim3_t dims) {
        // order: nslc, ncol, nrow
        int rank    = 1;
        int n[]     = {dims.z};
        int istride = 1;
        int ostride = 1;
        int idist   = dims.z;
        int odist   = dims.z;
        int batches = dims.x * dims.y;

        cufftHandle plan;
        cufftResult status = cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, batches);
        if (status != CUFFT_SUCCESS) {
            int dev = -1;
            cudaGetDevice(&dev);
            std::cerr << "Failed to make a plan on device " << dev << "." << std::endl;
            std::cerr << "Error code: " << status << std::endl;
            throw status;
        }
        return plan;
    }

    inline cufftHandle fftPlan2D(dim3_t dims) {
        // order: nslc, ncol, nrow
        int rank    = 2;
        int n[]     = {dims.y, dims.z};
        int istride = 1;
        int ostride = 1;
        int idist   = dims.y * dims.z;
        int odist   = dims.y * dims.z;
        int batches = dims.x;

        cufftHandle plan;
        cufftResult status = cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, batches);
        if (status != CUFFT_SUCCESS) {
            int dev = -1;
            cudaGetDevice(&dev);
            std::cerr << "Failed to make a plan on device " << dev << "." << std::endl;
            std::cerr << "Error code: " << status << std::endl;
            throw status;
        }
        return plan;
    }

    void fftshift_center(dev_arrayc, float, cudaStream_t);

    void ifftshift_center(dev_arrayc, float, cudaStream_t);

    void fftshift1D(dev_arrayc, cudaStream_t);

    void fftshift2D(dev_arrayc, cudaStream_t);

} // namespace tomocam
#endif // TOMOCAM_FFT__H
