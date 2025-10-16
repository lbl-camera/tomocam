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
         * @param axis axis to shift
         * @return shifted array
         */
        template <typename T>
        DeviceArray<T> roll(const DeviceArray<T> &, int3);

        /** fftshift
         * Shifts the zero-frequency component to the center of the array.
         *
         * @param in input array
         * @param shfit numper of pixels to shift
         *
         * @return shifted array
         */
        template <typename T>
        DeviceArray<complex_t<T>> phase_shift(const DeviceArray<complex_t<T>> &, T);
    } // namespace gpu
} // namespace tomocam
#endif // TOMOCAM_GPU_OPS_CUH
