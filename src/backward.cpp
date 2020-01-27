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
#include <fstream>

#include "dev_array.h"
#include "dist_array.h"
#include "fft.h"
#include "tomocam.h"
#include "types.h"

namespace tomocam {

    void backProject(float *input, float *output, dim3_t idims, dim3_t odims, float over_sampling, float center,
        DeviceArray<float> angles, kernel_t kernel, cudaStream_t stream) {

        // working dimensions
        size_t nelems = idims.x * idims.y * idims.z;
        size_t padded = (size_t)((float)idims.z * over_sampling);
        dim3_t pad_idims(idims.x, idims.y, padded);
        dim3_t pad_odims(odims.x, padded, padded);

        // data sizes
        size_t istreamSize = pad_idims.x * pad_idims.y * pad_idims.z;
        size_t ostreamSize = pad_odims.x * pad_odims.y * pad_odims.z;

        // buffers for input and output
        cuComplex_t *temp     = NULL;
        cuComplex_t *d_input  = NULL;
        cuComplex_t *d_output = NULL;

        cudaError_t status = cudaMalloc((void **)&temp, nelems * sizeof(cuComplex_t));
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to allocate memory device. " << status << std::endl;
            throw status;
        }
        status = cudaMalloc((void **)&d_input, istreamSize * sizeof(cuComplex_t));
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to allocate memory device. " << status << std::endl;
            throw status;
        }
        status = cudaMalloc((void **)&d_output, ostreamSize * sizeof(cuComplex_t));
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to allocate memory device. " << status << std::endl;
            throw status;
        }

        // set everything to zero, for padding. Don't expect this to throw exceptions
        cudaMemsetAsync(d_input, 0, istreamSize * sizeof(cuComplex_t), stream);
        cudaMemsetAsync(d_output, 0, ostreamSize * sizeof(cuComplex_t), stream);

        // copy data to streams (real -> complex)
        status = cudaMemcpy2DAsync(temp, sizeof(cuComplex_t), input, sizeof(float), sizeof(float),
            nelems, cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to copy F2C data to device. " << status << std::endl;
            throw status;
        }

        // pad data for oversampling
        int nrows = idims.x * idims.y;
        for (int i = 0; i < nrows; i++) {
            size_t offset1 = i * idims.z;
            size_t offset2 = i * padded;
            status         = cudaMemcpyAsync(
                d_input + offset2, temp + offset1, sizeof(cuComplex_t) * idims.z, cudaMemcpyDeviceToDevice, stream);
            if (status != cudaSuccess) {
                std::cerr << "Error! failed to copy data to device. " << status << std::endl;
                throw status;
            }
        }
        cudaStreamSynchronize(stream);

        // fftshift
        fftshift1D(d_input, pad_idims, stream);
        cudaStreamSynchronize(stream);

        // 1-D fft
        cufftHandle p1 = fftPlan1D(pad_idims);
        cufftSetStream(p1, stream);
        cufftResult error = cufftExecC2C(p1, d_input, d_input, CUFFT_FORWARD);
        if (error != CUFFT_SUCCESS) {
            std::cerr << "Error! failed to execute 1-D FWD Fourier transform. " << error << std::endl;
            throw error;
        }
        cudaStreamSynchronize(stream);
        cufftDestroy(p1);

        // center shift
        ifftshift_center(d_input, pad_idims, center, stream);
        cudaStreamSynchronize(stream);

        // covolution with kernel
        polarsample_transpose(d_input, d_output, pad_idims, pad_odims, angles, kernel, stream);
        cudaStreamSynchronize(stream);
   
        // fftshift
        fftshift2D(d_output, pad_odims, stream);
        cudaStreamSynchronize(stream);

        // 2-D ifft
        cufftHandle p2 = fftPlan2D(pad_odims);
        cufftSetStream(p2, stream);
        error = cufftExecC2C(p2, d_output, d_output, CUFFT_INVERSE);
        if (error != CUFFT_SUCCESS) {
            std::cerr << "Error! failed to execute 2-D INV Fourier transform. " << error << std::endl;
            throw error;
        }
        cudaStreamSynchronize(stream);
        cufftDestroy(p2);

        /*////// DEBUG
        size_t IMG = pad_odims.y * pad_odims.z;
        cuComplex_t * ff = new cuComplex_t[IMG];
        status = cudaMemcpyAsync(ff, d_output, IMG * sizeof(cuComplex_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::ofstream real("real.out", std::ios::out);
        std::ofstream imag("imag.out", std::ios::out);
        for (int i = 0; i < IMG; i++) {
            real << ff[i].x << " ";
            imag << ff[i].x << " ";
        }
        real.close();
        imag.close();
        exit(1);
        /////// */
   
        // fftshift
        fftshift2D(d_output, pad_odims, stream);
        cudaStreamSynchronize(stream);

        // rescale FFT(X) / N
        float scale = 1.f / ((float) (pad_odims.y * pad_odims.z));
        rescale(d_output, pad_odims, scale, stream);
        cudaStreamSynchronize(stream);
        
        // de-apodizing factor
        float W = 2 * kernel.radius() + 1;
        float beta = kernel.beta();
        deapodize(d_output, pad_odims, W, beta, stream);
        cudaStreamSynchronize(stream);

        // remove padding
        nelems = odims.x * odims.y * odims.z;
        cuComplex_t * temp2 = NULL;
        status = cudaMalloc((void **)&temp2, sizeof(cuComplex_t) * nelems);
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to allocate memory on device " << status << std::endl;
            throw status;
        }
        size_t ipad = (pad_odims.z - odims.z)/2;
        for (int i = 0; i < odims.x; i++)
            for (int j = 0; j < odims.y; j++) {
                size_t offset1 = i * odims.y * odims.z + j * odims.z;
                size_t offset2 = i * pad_odims.y * pad_odims.z + (j + ipad) * pad_odims.z + ipad;

                status = cudaMemcpyAsync(
                    temp2 + offset1, d_output + offset2, sizeof(cuComplex_t) * odims.z, cudaMemcpyDeviceToDevice, stream);
            }

        // copy data back to host
        status = cudaMemcpy2DAsync(
            output, sizeof(float), temp2, sizeof(cuComplex_t), sizeof(float), nelems, cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to copy C2F data from device. " << status << std::endl;
            throw status;
        }

        /*////// DEBUG
        size_t IMG = odims.y * odims.z;
        std::ofstream real("slice.out", std::ios::out);
        for (int i = 0; i < IMG; i++) {
            real << output[i] << " ";
        }
        real.close();
        ///// END DEBUG */

        // clean up
        cudaStreamSynchronize(stream);
        cudaFree(temp);
        cudaFree(temp2);
        cudaFree(d_input);
        cudaFree(d_output);
    }
} // namespace tomocam
