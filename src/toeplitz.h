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

#include "machine.h"
#include "dev_array.h"
#include "internals.h"
#include "fft.h"
#include "types.h"

#include <gpu/circshift.cuh>

#ifndef TOEPLITZ__H
#define TOEPLITZ__H

namespace tomocam {


    class SpreadFunc {
        private:
            DeviceArray<cuComplex_t> psf_;

        public: 
            SpreadFunc() = default;
            SpreadFunc(NUFFTGrid &grid, int nproj, int nrows) {

                int N1 = fftwsize(2 * nrows);
                dim3_t dims(1, nproj, nrows);

                // allocate ones
                auto ones = DeviceArray<cuComplex_t>(dims);
                ones.init(cuComplex_t(1.f,0), cudaStreamPerThread);

                // allocate psf
                auto temp = DeviceArray<cuComplex_t>(dim3_t(1, N1, N1));
        
                // compute nufft type 1
                nufft2d1(ones, temp, grid);

                // get the real part
                auto temp2 = cmplx_to_real(temp, cudaStreamPerThread);

                // center shift
                DeviceArray temp3 = temp2;
                gpu_ops::fftshift(temp3.dev_ptr(), temp2.dev_ptr(), N1, N1);

                // allocate space for the Fourier transform of the spread function
                dim3_t dims2 = {1, N1, N1/2+1};

                // allocate space for Fourier transformed signals
                psf_ = DeviceArray<cuComplex_t>(dims2);

                // compute FFT(psf)
                fft2D_r2c(temp3, psf_, cudaStreamPerThread);

                // normalize
                float w = 1.f; // / static_cast<float>(N1 * N1);
                rescale(psf_, w, cudaStreamPerThread);
            }

            DeviceArray<float> convolve(const DeviceArray<float> &x, cudaStream_t s) const {

                // zero pad to match psf size
                dim3_t dims = {x.dims().x, psf_.dims().y, psf_.dims().y};
                auto xpad = add_padding(x, dims, s);

                // allocate memory for Fourier-transformed array
                auto xft = DeviceArray<cuComplex_t>(psf_.dims());

                // fft(x) Real -> complex
                fft2D_r2c(xpad, xft, s);

                // multiply 
                auto xft_psf = xft.multiply(psf_, s);

                // ifft(g * x) complex -> real
                fft2D_c2r(xft_psf, xpad, s);

                // normalize fft
                float w = static_cast<float>(dims.y * dims.y);
                xpad.divide(w, s);
 
                // remove padding
                auto g = remove_padding(xpad, x.dims(), s);
                return g;
            }

    };
}

#endif // TOEPLITZ__H
