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
#include "kernel.h"
#include "fft.h"
#include "tomocam.h"
#include "internals.h"
#include "types.h"

namespace tomocam {

    void fwd_project(dev_arrayc input,  dev_arrayc output, float center,
            dev_arrayf angles, kernel_t kernel, cudaStream_t stream) {

        // fftshift
        fftshift2D(input, stream);
        cudaStreamSynchronize(stream);

        // 2-D forward fft
        cufftHandle p1 = fftPlan2D(input.dims());
        cufftSetStream(p1, stream);
        cufftResult error = cufftExecC2C(p1, input.dev_ptr(), input.dev_ptr(), CUFFT_FORWARD);
        if (error != CUFFT_SUCCESS) {
            std::cerr << "Error! failed to execute 1-D FWD Fourier transform. " << error << std::endl;
            throw error;
        }
        cudaStreamSynchronize(stream);
        cufftDestroy(p1);
   
        // fftshift
        fftshift2D(input, stream);
        cudaStreamSynchronize(stream);

        // rescale FFT(X) / N 
        rescale(input, stream);
        cudaStreamSynchronize(stream);

        // covolution with kernel
        polarsample(input, output, angles, kernel, stream);
        cudaStreamSynchronize(stream);

        // redo the center shift
        fftshift_center(output, center, stream);
        cudaStreamSynchronize(stream);

        // 1-D ifft
        cufftHandle p2 = fftPlan1D(output.dims());
        cufftSetStream(p2, stream);
        error = cufftExecC2C(p2, output.dev_ptr(), output.dev_ptr(), CUFFT_INVERSE);
        if (error != CUFFT_SUCCESS) {
            std::cerr << "Error! failed to execute 2-D INV Fourier transform. " << error << std::endl;
            throw error;
        }
        cudaStreamSynchronize(stream);
        cufftDestroy(p2);
   
        // fftshift
        fftshift1D(output, stream);
        cudaStreamSynchronize(stream);

        // de-apodizing factor
        deapodize1D(output, kernel, stream);
        cudaStreamSynchronize(stream);
    }
} // namespace tomocam
