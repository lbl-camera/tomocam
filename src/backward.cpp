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
#include "internals.h"
#include "types.h"

#include "debug.cuh"

namespace tomocam {

    void back_project(DeviceArray<cuComplex_t> d_input, DeviceArray<cuComplex_t> d_output, 
        float center, DeviceArray<float> angles, kernel_t kernel, cudaStream_t stream) {


        cuComplex_t * input = d_input.dev_ptr();
        dim3_t idims = d_input.dims();
        cuComplex_t * output = d_output.dev_ptr();
        dim3_t odims = d_output.dims();
        
        // fftshift
        fftshift1D(d_input, stream);
        cudaStreamSynchronize(stream);

        // 1-D fft
        cufftHandle p1 = fftPlan1D(idims);
        cufftSetStream(p1, stream);
        cufftResult error = cufftExecC2C(p1, input, input, CUFFT_FORWARD);
        if (error != CUFFT_SUCCESS) {
            std::cerr << "Error! failed to execute 1-D FWD Fourier transform. " << error << std::endl;
            throw error;
        }
        cudaStreamSynchronize(stream);
        cufftDestroy(p1);

        // rescale FFT(X) / N
        float scale = 1.f / ((float) (idims.z * idims.z));
        rescale(input, idims, scale, stream);
        cudaStreamSynchronize(stream);

        // center shift
        ifftshift_center(d_input, center, stream);
        cudaStreamSynchronize(stream);

        // covolution with kernel
        polarsample_transpose(d_input, d_output, angles, kernel, stream);
        cudaStreamSynchronize(stream);
 
        // fftshift
        fftshift2D(d_output, stream);
        cudaStreamSynchronize(stream);

        // 2-D ifft
        cufftHandle p2 = fftPlan2D(odims);
        cufftSetStream(p2, stream);
        error = cufftExecC2C(p2, output, output, CUFFT_INVERSE);
        if (error != CUFFT_SUCCESS) {
            std::cerr << "Error! failed to execute 2-D INV Fourier transform. " << error << std::endl;
            throw error;
        }
        cudaStreamSynchronize(stream);
        cufftDestroy(p2);

        // fftshift
        fftshift2D(d_output, stream);
        cudaStreamSynchronize(stream);
        
        // de-apodizing factor
        float W = 2 * kernel.radius() + 1;
        float beta = kernel.beta();
        deApodize2D(output, odims, W, beta, stream);
        cudaStreamSynchronize(stream);
    }
} // namespace tomocam
