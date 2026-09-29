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
#ifndef TOEPLITZ__H
#define TOEPLITZ__H

#include "dev_array.h"
#include "fft.h"
#include "gpu/padding.cuh"
#include "nufft.h"
#include "types.h"

namespace tomocam {

    // smallest N >= n of the form 2^a * 3^b -- cuFFT (and cufinufft's
    // internal FFTs) fall back to the much slower Bluestein algorithm for
    // sizes with large prime factors, so the Toeplitz PSF's zero-padded
    // convolution size should always be rounded up to one of these
    // "fast" lengths rather than used as-is.
    inline int next_fast_len(int n) {
        if (n <= 1) return 1;
        long long best = -1;
        for (long long p3 = 1;; p3 *= 3) {
            long long p2 = 1;
            while (p3 * p2 < n) p2 *= 2;
            long long cand = p3 * p2;
            if (best == -1 || cand < best) best = cand;
            if (p3 >= n) break; // larger p3 (with p2=1) only grows from here
        }
        return static_cast<int>(best);
    }

    template <typename T>
    class PointSpreadFunction {
      private:
        DeviceArray<gpu::complex_t<T>> psf_;

      public:
        PointSpreadFunction() = default;
        PointSpreadFunction(const nufft::Grid<T> &grid) {

            // compute the size of the psf
            int nproj = grid.nprojs();
            int ncols = grid.npixels();

            // L = 2*ncols-1 is the minimal support needed to exactly
            // represent A^TA's Toeplitz kernel for an ncols-sized object
            // (see convolve()'s comment for the derivation of why the
            // fixed crop offset of ncols-1 below stays correct however
            // much this gets padded further). Generate the psf at that
            // natural size first...
            int L = 2 * ncols - 1;

            // allocate ones
            constexpr gpu::complex_t<T> v(1, 0);
            DeviceArray<gpu::complex_t<T>> ones(dim3_t(1, nproj, ncols));
            ones.init(v);

            // compute nufft type 1
            dim3_t out_dims(1, L, L);
            DeviceArray<gpu::complex_t<T>> temp(out_dims);
            nufft::nufft2d1(ones, temp, grid);

            // get the real part
            auto psf = to_real<T>(temp);

            // ...then zero-pad (offset 0, i.e. PadType::RIGHT, matching
            // convolve()'s padding of x) up to the next FFT-fast length
            // before transforming, so the convolution below runs cuFFT's
            // fast path instead of Bluestein.
            int N1 = next_fast_len(L);
            if (N1 > L) psf = gpu::pad2d<T>(psf, N1 - L, PadType::RIGHT);

            // compute FFT(psf)
            psf_ = rfft2D(psf);
        }

        // Peak GPU memory convolve() needs per slice. convolve() zero-pads
        // each slice from ncols x ncols up to the PSF's N1 x N1 before the
        // FFT-based convolution, and keeps two padded real buffers (xpad,
        // tmp2) and two padded complex buffers (xft, xft_psf) alive at
        // once -- callers that chunk their input before calling convolve()
        // (gradient2_(), funcval2()) should size chunks against this, not
        // against the plain input array's bytes, which undercounts by the
        // padding/multi-buffer factor (~16-17x for typical N1 ~= 2*ncols).
        size_t convolve_peak_bytes_per_slice() const {
            size_t N1 = psf_.nrows();
            size_t N1c = psf_.ncols(); // N1/2 + 1, from the rfft2D output
            size_t real_buf = N1 * N1 * sizeof(T);                     // xpad, tmp2
            size_t complex_buf = N1 * N1c * sizeof(gpu::complex_t<T>); // xft, xft_psf
            return 2 * real_buf + 2 * complex_buf;
        }

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

            // Crop the valid ncols x ncols window out of the N1 x N1
            // circular convolution. x sits at offset 0 (PadType::RIGHT
            // above) and psf's support sits at offset 0 too (see ctor),
            // covering indices [0, 2*ncols-2]. For output index k in
            // [0, ncols), circular index (k + ncols-1 - j) for j in
            // [0, ncols) ranges over exactly [0, 2*ncols-2] -- i.e. it
            // never wraps around mod N1 for ANY N1 >= 2*ncols-1, not just
            // N1 == 2*ncols-1. So the correct window always starts at the
            // FIXED offset ncols-1, regardless of how much fast-length
            // padding N1 adds beyond the minimal 2*ncols-1 -- it is not
            // N1 - ncols (that only coincided with ncols-1 when
            // N1 == 2*ncols-1 exactly), hence crop2d with an explicit
            // shift instead of unpad2d's size-difference-derived one.
            int shift = x.nrows() - 1;
            auto g = gpu::crop2d<T>(tmp2, x.nrows(), shift);

            return g / (scale1 * scale2);
        }
    };
} // namespace tomocam

#endif // TOEPLITZ__H
