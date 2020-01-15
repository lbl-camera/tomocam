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

#include "dev_array.h"
#include "dist_array.h"
#include "tomocam.h"
#include "types.h"

namespace tomocam {

    void backproject(
        Partition<float> input, 
        Partition<float> output, 
        float over_sampling,
        DeviceArray<float2> polar_coords,
        kernel_t kernel, 
        cudaStream_t stream){

        // dims
        dim3_t idims = input.dims();
        dim3_t odims = output.dims();

        // pointers to data buffers
        float *h_input_buf = input.begin();
        float *h_output_buf = output.begin();

        // working dimensions
        size_t nelems = idims.x * idims.y * idims.z;
        size_t padded = (size_t) ((float) idims.z * over_sampling); 
        dim3_t pidims(idims.x, idims.y, padded);

        size_t padrows = (size_t) ((float) odims.y * over_sampling);
        size_t padcols = (size_t) ((float) odims.z * over_sampling);
        dim3_t podims(odims.x, padrows, padcols);

        // data sizes
        size_t istreamSize = pidims.x * pidims.y * pidims.z;
        size_t ostreamSize = podims.x * podims.y * podims.z;

        // buffers for input and output
        cuComplex_t *temp     = NULL;
        cuComplex_t *d_input  = NULL;
        cuComplex_t *d_output = NULL;

        
        status = cudaMalloc((void **) &temp, nelems * sizeof(cuComplex_t));
        status = cudaMalloc((void **) &d_input, istreamSize * sizeof(cuComplex_t));
        status = cudaMalloc((void **) &d_output, ostreamSize * sizeof(cuComplex_t));

        status = cudaMemset(d_input, 0, nelems * sizeof(cuComplex_t));
        status = cudaMemset(d_input, 0, istreamSize * sizeof(cuComplex_t));
        status = cudaMemset(d_output, 0, ostreamSize * sizeof(cuComplex_t));

        // copy data to streams
        status = cudaMemcpy2DAsync(
                    temp,
                    sizeof(cuComplex_t), 
                    h_input_buf, 
                    sizeof(float),
                    sizeof(float),
                    istreamSize,
                    cudaMemcpyHostToDevice,
                    stream);
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to copy data to device. " << status << std::endl;
            throw status;
        }

        // pad data for oversampling
        int nrows = idims.x * idims.y;
        for (int i = 0; i < nrows; i++){
            size_t offset1 = i * idims.z;
            size_t offset2 = i * padded;
            status = cudaMemcpyAsync(
                        d_input + offset2,
                        temp + offset1,
                        sizeof(cuComplex_t) * idims.z,
                        cudaMemcpyDeviceToDevice,
                        stream);
            if (status != cudaSuccess) {
                std::cerr << "Error! failed to copy data to device. " << status << std::endl;
                throw status;
            }
        }
        cudaFree(temp);

        // fftshift
        fftshift1D(d_input, pidims, stream);

        // 1-D fft
        cufftHandle p = fftPlan1D(pidims);
        cufftSetStream(p, stream);
        cufftExecC2C(p, d_input, d_input, CUFFT_FORWARD) 
        cufftDestroy(p);
        
        // center shift
        fftshift_center(d_input, pidims, center, stream);

        // covolution with kernel
        polarsample_transpose(d_input, d_output, pidims, podims, kernel, polar_coords, stream);

        // 2-D ifft
        p = fftPlan2D(podims);
        cufftSetStream(p, stream);
        cufftExecC2C(p, d_output, d_output, CUFFT_INVERSE);
        cufftDestroy(p);

        // fftshift
        fftshift2D(d_output, podims, stream);

        // Apodizing factor
        apodization_correction(d_output, podims, stream);

        // remove padding
        nelems = odims.x * odims.y * odims.z;
        status = cudaMalloc((void **) &temp, sizeof(cuComplex_t) * nelems);
        for (int i = 0; i < odims.x; i++)
            for (int j = 0; i < odims.y; j++) {
                size_t offset1 = i * odims.y * odims.z + j * odims.z;
                size_t offset2 = i * podims.y * podims.z + j * podims.z;
                status = cudaMemcpyAsync(
                            temp + offset1, 
                            d_output + offset2, 
                            sizeof(cuComplex_t) * odims.z, 
                            cudaMemcpyDeviceToDevice, 
                            stream);
            }

        // copy data back to host
        status = cudaMemcpy2DAsync(
                    h_output_buf,
                    sizeof(float), 
                    temp, 
                    sizeof(cuComplex_t), 
                    sizeof(float), 
                    ostreamSize, 
                    cudaMemcpyDeviceToHost, 
                    stream);
    }
} // namespace tomocam
