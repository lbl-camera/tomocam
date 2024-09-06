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
        DeviceArray<gpu::complex_t<T>> psf_;

      public:
        PointSpreadFunction() = default;

        PointSpreadFunction(const NUFFT::Grid<T> &grid) {

            // set the device
            device_ = grid.dev_id();

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

            // zero pad for convolution (N1 + ncols - 1)
            psf = gpu::pad2d<T>(psf, ncols - 1, PadType::RIGHT);

            // compute FFT(psf)
            psf_ = rfft2D(psf);
        }

        DeviceArray<T> convolve(const DeviceArray<T> &x) const {

            // set the device
            int dev;
            SAFE_CALL(cudaGetDevice(&dev));
            if (dev != device_) {
                std::cerr
                    << "Error: device mismatch in PointSpreadFunction::convolve"
                    << std::endl;
            }

            // pad x to match the size of the psf
            int padding = psf_.nrows() - x.nrows();

            // zero pad
            auto xpad = gpu::pad2d<T>(x, padding, PadType::RIGHT);

            // fft(x) Real -> complex
            auto xft = rfft2D<T>(xpad);

            // broadcast-multiply
            auto xft_psf = xft.multiply(psf_);

            // scale by the size of the image
            xft_psf /= static_cast<T>(xpad.nrows() * xpad.ncols());

            // ifft(g * x) complex -> real
            auto tmp2 = irfft2D<T>(xft_psf);

            // remove padding
            return gpu::unpad2d<T>(tmp2, padding);
        }
    };
}

#endif // TOEPLITZ__H
