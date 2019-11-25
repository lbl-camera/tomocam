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



#include <cuda_runtime.h>
#include <cufft.h>

#include "machine.h"
#include "fft.h"
#include "util.h"

const int nStreams = 3;
const int slcsPerStream = 10;

namespace tomocam {
    cufftHandle fftPlan1D(int dims[]) {
        // order: nslc, ncol, nrow
        int rank    = 1;
        int n[]     = {dims[2]};
        int istride = 1;
        int ostride = 1;
        int idist   = dims[2];
        int odist   = dims[2];
        int batches = dims[0] * dims[1];
        cufftHandle plan;
        if (cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride,
                          odist, CUFFT_C2C, batches) != CUFFT_SUCCESS) {
            fprintf(stderr, "Failed to create a 1-d plan :-(.\n");
            return NULL;
        }
        return plan;
    }

    cufftHandle fftPlan2D(int dims[]) {
        // order: nslc, ncol, nrow
        int rank      = 2;
        int n[]       = {dims[1], dims[2]};
        int inembed[] = {dims[1], dims[2]};
        in  onembed[] = {dims[1], dims[2]};
        int istride   = 1;
        int ostride   = 1;
        int idist     = dims[1] * dims[2];
        int odist     = dims[1] * dims[2];
        int batches   = dims[0];

        cufftHandle plan;
        if (cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed,
                          ostride, odist, CUFFT_C2C,
                          batches) != CUFFT_SUCCESS) {
            fprintf(stderr, "Failed to create a 1-d plan :-(.\n");
            return NULL;
        }
        return plan;
    }

    
    void DArrayFFT(Partition<complex_t> &input, Partition<complex_t> &output,
                    int FFT_DIRECTION, bool is_1D) {

        std::vecotr<cufftHandle> plans;
        std::vecotr<cudaStream_t> streams;

        dim3_t pDims = input.dims();
        int dims[] = { slcsPerStream, pDims.y, pDims.z };
        size_t streamSize = dims[0] * dims[1] * dims[2];
        size_t streamBytes = streamBytes * sizeof(complex_t);

        // create cuda streams
        for (int i = 0; i < nStreams; i++) {
            cudaStream_t temp;
            cudaStreamCreate(&temp);
            streams.push_back(temp);
        }


        if (is_1D) {
            for (int i = 0; i < nStreams; i++) {
                auto p = fftPlan1D(3, dims);
                cudaSetStream(p, streams[i]);
                plans.push_back(p);
            }
        } else {
            for (int i = 0; i < nStreams; i++) {
                auto p = fftPlan2D(3, dims);
                cudaSetStream(p, streams[i]);
                plans.push_back(p);
            }
        }

        // host data from partiion
        complex_t * h_data = input.begin();

        // device data buffer
        complex_t * d_data = cudaMalloc((void **) &d_data, nStreams * streamBytes);

        unsigned nIters = pDims.x / (nStreams * slcsPerStream);
        for (int it = 0; it < nIters; it++) {
    	    for (int i = 0; i < nStreams; ++i) {
        	    int offset = i * streamSize;
        	    cudaMemcpyAsync(d_data + offset, h_data + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);
    	    }
    	    // FFT of a signal
    	    for (int i = 0; i < nStreams; ++i) {
        	    int offset = i * streamSize;
        	    status = cufftExecC2C(plans[i], d_data + offset, d_data + offset, FFT_DIRECTION);
        	    if (status != CUFFT_SUCCESS) {
            	    cout << "Cufft FFT work error: " << status << endl;
        	    }
    	    }
		    // copy data back to host
		    for (int i = 0; i < nStreams; i++) {
        	    int offset = i * streamSize;
        	    cudaMemcpyAsync(h_data + offset, d_Data + offset, streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    	    }
            h_data += nStreams * streamSize;
        }
        // destroy plans
        for (auto p : plans) cufftDestroy(p);

        // do the remaing work
        unsigned nExtra = pDims.x % (nStreams * slcsPerStream);
        std::vecotr<unsigned> ranks = distribute(nExtra, nStreams);
        for (int i = 0; i < nStreams; i++) {
            dims[0] = ranks[i];
            p = fftPlan1D(3, dims);
            cudaSetStream(p, streams[i]);
            plans.push_back(p);
        }
        // copy remaining data to device
        int offset = 0;
        for (int i = 0; i < nStreams; i++) {
            streamSize = ranks[i] * dims[1] * dims[2];
        	cudaMemcpyAsync(d_data + offset, h_data + offset, streamSize * sizeof(complex_t), cudaMemcpyHostToDevice, streams[i]);
            offset += streamSize;
    	}
        // do the transform
        offset = 0;
    	for (int i = 0; i < nStreams; ++i) {
        	status = cufftExecC2C(plans[i], d_data + offset, d_data + offset, CUFFT_FORWARD);
        	if (status != CUFFT_SUCCESS) {
                std::cout << "Cufft FFT work error: " << status << endl;
        	}
            offset += ranks[i] * dims[1] * dims[2];
    	}
		// copy data back to host
        offset = 0;
		for (int i = 0; i < nStreams; i++) {
            streamSize = ranks[i] * dims[1] * dims[2];
            cudaMemcpyAsync(h_data + offset, d_Data + offset, streamSize * sizeof(complex_t), cudaMemcpyDeviceToHost, streams[i]);
            offset += streamSize;
    	}
          
        //clean up
        for (auto s : streams) cudaStreamSynchronize(s);
        for (auto s : streams) cudaStreamDestroy(s);
        for (auto p : planB) cufftDestroy(p);
        cudaFree(d_data);
    }

    void fft1d (DArray<complex_t> & input, DArray<float> & output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        #pragma parallel for num_threads(nDevice)
        for (int i = 0; i < p1.size(); i++) {
            unsigned device = i % nDevice;
            cudaSetDevice(device);
            DArrayFFT(p1[i], p2[i], CUFFT_FORWARD, true);
        }
        cudaDeviceSynchronize();
    }

    void ifft1d (DArray<complex_t> & input, DArray<float> & output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        #pragma parallel for num_threads(nDevice)
        for (int i = 0; i < p1.size(); i++) {
            unsigned device = i % nDevice;
            cudaSetDevice(device);
            DArrayFFT(p1[i], p2[i], CUFFT_INVERSE, true);
        }
        cudaDeviceSynchronize();
    }

    void fft2d (DArray<complex_t> & input, DArray<float> & output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        #pragma parallel for num_threads(nDevice)
        for (int i = 0; i < p1.size(); i++) {
            unsigned device = i % nDevice;
            cudaSetDevice(device);
            DArrayFFT(p1[i], p2[i], CUFFT_FORWARD, false);
        }
        cudaDeviceSynchronize();
    }

    void ifft2d (DArray<complex_t> & input, DArray<float> & output) {
        int nDevice = MachineConfig::getInstance().num_of_gpus();
        std::vector<Partition<complex_t>> p1 = input.create_partitions(nDevice);
        std::vector<Partition<complex_t>> p2 = output.create_partitions(nDevice);

        #pragma parallel for num_threads(nDevice)
        for (int i = 0; i < p1.size(); i++) {
            unsigned device = i % nDevice;
            cudaSetDevice(device);
            DArrayFFT(p1[i], p2[i], CUFFT_INVERSE, false);
        }
        cudaDeviceSynchronize();
    }
} // namespace tomocam
