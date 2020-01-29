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
#include "kernel.h"
#include "dist_array.h"
#include "fft.h"
#include "tomocam.h"
#include "internals.h"
#include "types.h"

namespace tomocam {

    void stage_fwd_project(float *input, float *output, dim3_t idims, dim3_t odims, float over_sampling, float center,
        DeviceArray<float> angles, kernel_t kernel, cudaStream_t stream) {

        // working dimensions
        size_t nelems = idims.x * idims.y * idims.z;
        size_t padded = (size_t)((float)idims.z * over_sampling);
        dim3_t pad_idims(idims.x, padded, padded);
        dim3_t pad_odims(odims.x, odims.y, padded);

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
        int ipad = (padded - idims.z) / 2;
        for (int i = 0; i < idims.x; i++)
            for (int j = 0; j < idims.y; j++) {
                size_t offset1 = i * pad_idims.y * pad_idims.z + (j + ipad) * pad_idims.z + ipad;
                size_t offset2 = i * idims.y * idims.z + j * idims.z;

                status = cudaMemcpyAsync(
                    d_input + offset1, temp + offset2, sizeof(cuComplex_t) * idims.z, cudaMemcpyDeviceToDevice, stream);
                if (status != cudaSuccess) {
                    std::cerr << "Error! failed to copy data to device. " << status << std::endl;
                    throw status;
                }
            }
        cudaStreamSynchronize(stream);

        // do the actual forward projection
        fwd_project(d_input, d_output, pad_idims, pad_odims, center, angles, kernel, stream);

        // remove padding
        nelems = odims.x * odims.y * odims.z;
        cuComplex_t * temp2 = NULL;
        status = cudaMalloc((void **)&temp2, sizeof(cuComplex_t) * nelems);
        if (status != cudaSuccess) {
            std::cerr << "Error! failed to allocate memory on device " << status << std::endl;
            throw status;
        }
        for (int i = 0; i < odims.x * odims.y; i++) {
            size_t offset1 = i * odims.y * odims.z;
            size_t offset2 = i * pad_odims.y * pad_odims.z;

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

        // clean up
        cudaStreamSynchronize(stream);
        cudaFree(temp);
        cudaFree(temp2);
        cudaFree(d_input);
        cudaFree(d_output);
    }
} // namespace tomocam
