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

#include <cuda/std/complex>

#include "dev_array.h"

#ifndef TOMOCAM_GPU_OPS_CUH 
#define TOMOCAM_GPU_OPS_CUH

namespace tomocam {
    namespace gpu {

        /** roll
         * Shifts the elements of the array by a specified number of positions.
         *
         * @param in input array
         * @param shift number of pixels to shift
         * @return shifted array
         */
        template <typename T>
        DeviceArray<T> roll(const DeviceArray<T> &, int);

        /** roll2
         * Shifts the the elements of the array by a specified number of positions
         *
         * @param in input array
         * @param shfit numper of pixels to shift
         *
         * @return shifted array
         */
        template <typename T>
        DeviceArray<T> roll2(const DeviceArray<T> &, int, int);


        /** fftshift
         * Shifts the zero-frequency component to the center of the spectrum.
         *
         * @param in input array
         * @return shifted array
         */
        template <typename T>
        DeviceArray<T> fftshift(const DeviceArray<T> &in) {
            auto shfit = in.ncols() / 2;
            return roll(in, -shfit);
        }

        template <typename T>
        DeviceArray<T> fftshift2(const DeviceArray<T> &in) {
            auto shfit = in.ncols() / 2;
            auto shfit2 = in.nrows() / 2;
            return roll2(in, -shfit, -shfit2);
        }

        template <typename T>
        DeviceArray<T> ifftshift(const DeviceArray<T> &in) {
            auto shfit = in.ncols() / 2;
            return roll(in, shfit);
        }

        template <typename T>
        DeviceArray<T> ifftshift2(const DeviceArray<T> &in) {
            auto shfit = in.ncols() / 2;
            auto shfit2 = in.nrows() / 2;
            return roll2(in, shfit, shfit2);
        }

    } // namespace gpu
} // namespace tomocam
#endif // TOMOCAM_GPU_OPS_CUH
