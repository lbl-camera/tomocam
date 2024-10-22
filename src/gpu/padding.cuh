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

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef GPU_OPS_PADDING_CUH
#define GPU_OPS_PADDING_CUH

namespace tomocam {

    enum class PadType { LEFT, RIGHT, SYMMETRIC };
    namespace gpu {

        /**
         * @brief Pads a 1D array with zeros.
         * if pad_size is negative, the array is left padded, else right padded.
         *
         * @tparam T
         * @param arr
         * @param pad_size
         * @param PadType (LEFT, RIGHT, SYMMETRIC)
         * @return DeviceArray<T>
         */
        template <typename T>
        DeviceArray<T> pad1d(const DeviceArray<T> &, int, PadType);

        /**
         * @brief Unpads a 1D array.
         * if pad_size is negative, the array is left cropped, else right
         * cropped.
         *
         * @tparam T
         * @param arr
         * @param pad_size
         * @param PadType (LEFT, RIGHT, SYMMETRIC)
         * @return DeviceArray<T>
         */
        template <typename T>
        DeviceArray<T> unpad1d(const DeviceArray<T> &, int, PadType);

        /**
         * @brief pads a 2D array.
         * data is equally padded on both sides of the array.
         *
         * @tparam T
         * @param arr
         * @param pad_size
         * @param PadType (LEFT, RIGHT, SYMMETRIC)
         * @param stream
         * @return DeviceArray<T>
         */
        template <typename T>
        DeviceArray<T> pad2d(const DeviceArray<T> &, int, PadType);

        /**
         * @brief Unpads a 2D array.
         * data is equally cropped from both sides of the array.
         *
         * @tparam T
         * @param arr
         * @param pad_size
         * @param PadType (LEFT, RIGHT, SYMMETRIC)
         * @return DeviceArray<T>
         */
        template <typename T>
        DeviceArray<T> unpad2d(const DeviceArray<T> &, int, PadType);

    } // namespace gpu
} // namespace tomocam

#endif // GPU_OPS_PADDING_CUH
