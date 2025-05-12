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
#include "fft.h"
#include "internals.h"
#include "types.h"

#include "gpu/padding.cuh"

#ifndef TOEPLITZ__H
#define TOEPLITZ__H

namespace tomocam {

    template <typename T>
    class PointSpreadFunction {
        private:
            int device_;
            int batch_size_;
            bool initialized_ = false;
            cufftHandle r2c_;
            cufftHandle c2r_;
            DeviceArray<gpu::complex_t<T>> psf_;
            mutable DeviceArray<T> xpad_;
            
        public:

            PointSpreadFunction() = default;

            PointSpreadFunction(const NUFFT::Grid<T> &grid) {

                // set the device
                device_ = grid.dev_id();
                SAFE_CALL(cudaSetDevice(device_));

                int nproj = grid.nprojs();
                int ncols = grid.npixels();
                int N1 = 2 * ncols - 1;

                // allocate ones
                constexpr gpu::complex_t<T> v(1, 0);
                DeviceArray<gpu::complex_t<T>> ones(dim3_t(1, nproj, ncols));
                ones.init(v);

                // compute nufft type 1
                dim3_t out_dims(1, N1, N1);
                auto temp = NUFFT::nufft2d1(ones, grid, out_dims);

                // get the real part
                auto psf = real(temp);

                // compute FFT(psf)
                psf_ = rfft2D(psf);
            }

            ~PointSpreadFunction() {
                // destroy plans
                if (initialized_) {
                    SAFE_CALL(cufftDestroy(r2c_));
                    SAFE_CALL(cufftDestroy(c2r_));
                }
            }

            void create_plans(int batch_size) {
                // create plans
                auto dims = dim3_t(batch_size, psf_.nrows(), psf_.nrows());
                batch_size_ = batch_size;
                r2c_ = fftPlan2D(dims, CUFFT_R2C);
                c2r_ = fftPlan2D(dims, CUFFT_C2R);
                xpad_ = DeviceArray<T>(dims);
                // set the initialized flag
                initialized_ = true;
            }

            int batch_size() const { return batch_size_; }

            DeviceArray<T> convolve(const DeviceArray<T> &x) const {

                // scale for normalization
                T scale1 = std::pow(x.nrows(), 3);

                // pad x to match the size of the psf
                int padding = psf_.nrows() - x.nrows();
        
                // zero pad
                auto xpad = gpu::pad2d<T>(x, padding, PadType::RIGHT);
                T scale2 = static_cast<T>(xpad.nrows() * xpad.ncols());
             
                // fft(x) Real -> complex
                auto xft = rfft2D<T>(xpad);

                // broadcast-multiply
                auto xft_psf = xft.multiply(psf_);

                // ifft(g * x) complex -> real
                auto tmp2 = irfft2D<T>(xft_psf);
              
                // remove padding
                auto g = gpu::unpad2d<T>(tmp2, padding, PadType::LEFT);

                return g / (scale1 * scale2);
            }

            DeviceArray<T> convolve2(const DeviceArray<T> &x) const {

                // scale for normalization
                T scale1 = std::pow(x.nrows(), 3);

                // pad x to match the size of the psf
                int padding = psf_.nrows() - x.nrows();
        
                // zero pad
                gpu::pad2d<T>(xpad_, x, padding, PadType::RIGHT);
                T scale2 = static_cast<T>(xpad_.nrows() * xpad_.ncols());

                // create plans if not already created
                if (!initialized_) {
                    std::runtime_error("Error: plans not created in PointSpreadFunction::convolve");
                }
             
                // fft(x) Real -> complex
                auto xft = rfft2D<T>(r2c_, xpad_);

                // broadcast-multiply
                auto xft_psf = xft.multiply(psf_);

                // ifft(g * x) complex -> real
                auto tmp2 = irfft2D<T>(c2r_, xft_psf);
              
                // remove padding
                auto g = gpu::unpad2d<T>(tmp2, padding, PadType::LEFT);

                return g / (scale1 * scale2);
            }
    };
}

#endif // TOEPLITZ__H
