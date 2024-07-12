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


#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

#include "dist_array.h"
#include "dev_array.h"
#include "types.h"

#ifndef TOMOCAM_FFT__H
#define TOMOCAM_FFT__H

#define SAFE_CUFFT_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cufftResult code, const char *file, int line, bool abort=true){
   if (code != CUFFT_SUCCESS) {
      fprintf(stderr,"GPUassert: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

namespace tomocam {

    inline cufftHandle fftPlan1D(dim3_t dims, cufftType type) {
        // order: nslc, ncol, nrow
        int rank    = 1;
        int n[]     = {dims.z};
        int istride = 1;
        int ostride = 1;
        int idist   = dims.z;
        int odist   = dims.z;
        int batches = dims.x * dims.y;

        cufftHandle plan;
        SAFE_CUFFT_CALL(cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, type, batches));
        return plan;
    }

    inline cufftHandle fftPlan2D(dim3_t dims, cufftType type) {
        // order: nslc, ncol, nrow
        int rank    = 2;
        int n[]     = {dims.y, dims.z};
        int istride = 1;
        int ostride = 1;
        int idist   = dims.y * dims.z;
        int odist   = dims.y * dims.z;
        int batches = dims.x;

        cufftHandle plan;
        SAFE_CUFFT_CALL(cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, type, batches));
        return plan;
    }


    // execute cufft plan for single precision complex data
    inline void fftExec(cufftHandle plan, DeviceArraycf &in, DeviceArraycf &out,
        int dir, cudaStream_t s) {

        // attach stream
        SAFE_CUFFT_CALL(cufftSetStream(plan, s));

        // execute plan
        cufftComplex *idata = reinterpret_cast<cufftComplex *>(in.dev_ptr());
        cufftComplex *odata = reinterpret_cast<cufftComplex *>(out.dev_ptr());
        SAFE_CUFFT_CALL(cufftExecC2C(plan, idata, odata, dir));

        // wait for it to finish
        SAFE_CALL(cudaStreamSynchronize(s));
    }

    // execute cufft plan for double precision complex data
    inline void fftExec(cufftHandle plan, DeviceArraycd &in, DeviceArraycd &out,
        int dir, cudaStream_t s) {

        // attach stream
        SAFE_CUFFT_CALL(cufftSetStream(plan, s));

        // execute plan
        cufftDoubleComplex *idata = reinterpret_cast<cufftDoubleComplex *>(in.dev_ptr());
        cufftDoubleComplex *odata = reinterpret_cast<cufftDoubleComplex *>(out.dev_ptr());
        SAFE_CUFFT_CALL(cufftExecZ2Z(plan, idata, odata, dir));

        // wait for it to finish
        SAFE_CALL(cudaStreamSynchronize(s));
    }

    /** 
     * @brief Perform 1D forward FFT on a device array
     * 
     * @param in input array
     * @param s cuda stream
     * @return complex valued DeviceArray
     */
    template <typename T>
    inline DeviceArray<T> fft1D(DeviceArray<T> &in, cudaStream_t s) {

        // create a plan for 1D forward FFT
        if (std::is_same<T, gpu::complex_t<float>>::value) {
            cufftHandle plan = fftPlan1D(in.dims(), CUFFT_C2C);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, CUFFT_FORWARD, s);
            return out;
        } else if (std::is_same<T, gpu::complex_t<double>>::value) {
            cufftHandle plan = fftPlan1D(in.dims(), CUFFT_Z2Z);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, CUFFT_FORWARD, s);
            return out;
        } else {
            throw std::runtime_error("Unsupported data type");
        }
    } 


    /** 
     * @brief Perform 1D inverse FFT on a device array
     * 
     * @param in input array
     * @param s cuda stream
     * @return complex valued DeviceArray
     */
    template <typename T>
    inline DeviceArray<T> ifft1D(DeviceArray<T> &in, cudaStream_t s) {

        // create a plan for 1D inverse FFT
        if (std::is_same<T, gpu::complex_t<float>>::value) {
            cufftHandle plan = fftPlan1D(in.dims(), CUFFT_C2C);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, CUFFT_INVERSE, s);
            return out;
        } else if (std::is_same<T, gpu::complex_t<double>>::value) {
            cufftHandle plan = fftPlan1D(in.dims(), CUFFT_Z2Z);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, CUFFT_INVERSE, s);
            return out;
        } else {
            throw std::runtime_error("Unsupported data type");
        }
    }

    /** 
     * @brief Perform 2D forward FFT on a device array
     * 
     * @param in input array
     * @param s cuda stream
     * @return complex valued DeviceArray
     */
    template <typename T>
    inline DeviceArray<T> fft2D(DeviceArray<T> &in, cudaStream_t s) {

        // create a plan for 2D forward FFT
        if (std::is_same<T, gpu::complex_t<float>>::value) {
            cufftHandle plan = fftPlan2D(in.dims(), CUFFT_C2C);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, s);
            return out;
        } else if (std::is_same<T, gpu::complex_t<double>>::value) {
            cufftHandle plan = fftPlan2D(in.dims(), CUFFT_Z2Z);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, s);
            return out;
        } else {
            throw std::runtime_error("Unsupported data type");
        }
    }


    /** 
     * @brief Perform 2D inverse FFT on a device array
     * 
     * @param in input array
     * @param s cuda stream
     * @return complex valued DeviceArray
     */
    template <typename T>
    inline DeviceArray<T> ifft2D(DeviceArray<T> &in, cudaStream_t s) {

        // create a plan for 2D inverse FFT
        if (std::is_same<T, gpu::complex_t<float>>::value) {
            cufftHandle plan = fftPlan2D(in.dims(), CUFFT_C2C);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, CUFFT_INVERSE, s);
            return out;
        } else if (std::is_same<T, gpu::complex_t<double>>::value) {
            cufftHandle plan = fftPlan2D(in.dims(), CUFFT_Z2Z);
            
            // create output array
            DeviceArray<T> out(in.dims());
            fftExec(plan, in, out, CUFFT_INVERSE, s);
            return out;
        } else {
            throw std::runtime_error("Unsupported data type");
        }
    }


    /** @brief Perform a 2D Forward FFT on a real valued array
      * @param in input array 
      * @param s cuda stream
      * @return complex valued DeviceArray
      */
    template <typename T>
    DeviceArray<gpu::complex_t<T>> rfft2D(DeviceArray<T> &in, cudaStream_t s) {

        // create output array
        dim3_t dims = {in.nslices(), in.nrows(), in.ncols()/2+1};
        DeviceArray<gpu::complex_t<T>> out(dims);

        if (std::is_same<T, float>::value) {
            // create a plan
            cufftHandle plan = fftPlan2D(in.dims(), CUFFT_R2C);
            SAFE_CUFFT_CALL(cufftSetStream(plan, s));
            cufftReal *idata = reinterpret_cast<cufftReal *>(in.dev_ptr());
            cufftComplex *odata =
                reinterpret_cast<cufftComplex *>(out.dev_ptr());
            SAFE_CUFFT_CALL(cufftExecR2C(plan, idata, odata));
        } else if (std::is_same<T, double>::value) {
            cufftHandle plan = fftPlan2D(in.dims(), CUFFT_D2Z);
            SAFE_CUFFT_CALL(cufftSetStream(plan, s));
            cufftDoubleReal *idata =
                reinterpret_cast<cufftDoubleReal *>(in.dev_ptr());
            cufftDoubleComplex *odata =
                reinterpret_cast<cufftDoubleComplex *>(out.dev_ptr());
            SAFE_CUFFT_CALL(cufftExecD2Z(plan, idata, odata));
        } else { 
            throw std::runtime_error("Unsupported data type");
        }

        // wait for it to finish
        SAFE_CALL(cudaStreamSynchronize(s));
        return out;
    }

    /** Perform a 2D Inverse FFT on a complex valued array
      * @param in input array
      * @param s cuda stream
      * @return real valued DeviceArray
      */
    template <typename T>
    DeviceArray<T> irfft2D(DeviceArray<gpu::complex_t<T>> &in, cudaStream_t s) {

        // create output array, reconstruction is always going to be a square
        // matrix
        dim3_t dims = {in.nslices(), in.nrows(), in.nrows()};
        DeviceArray<T> out(dims);

        if (std::is_same<T, float>::value) {
            cufftHandle plan = fftPlan2D(dims, CUFFT_C2R);
            SAFE_CUFFT_CALL(cufftSetStream(plan, s));
            cufftComplex *idata =
                reinterpret_cast<cufftComplex *>(in.dev_ptr());
            cufftReal *odata = reinterpret_cast<cufftReal *>(out.dev_ptr());
            SAFE_CUFFT_CALL(cufftExecC2R(plan, idata, odata));
        } else if (std::is_same<T, double>::value) {
            cufftHandle plan = fftPlan2D(dims, CUFFT_Z2D);
            SAFE_CUFFT_CALL(cufftSetStream(plan, s));
            cufftDoubleComplex *idata =
                reinterpret_cast<cufftDoubleComplex *>(in.dev_ptr());
            cufftDoubleReal *odata =
                reinterpret_cast<cufftDoubleReal *>(out.dev_ptr());
            SAFE_CUFFT_CALL(cufftExecZ2D(plan, idata, odata));
        } else { 
            throw std::runtime_error("Unsupported data type");
        }

        // wait for it to finish
        SAFE_CALL(cudaStreamSynchronize(s));
        return out;
    }

} // namespace tomocam
#endif // TOMOCAM_FFT__H
