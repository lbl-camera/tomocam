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

#include "gpu/dev_memory.cuh"
#include "gpu/utils.cuh"

#include "dev_array.h"

namespace tomocam {
    namespace gpu {

        template <typename T>
        __global__ void roll_kernel(const DeviceMemory<T> in, DeviceMemory<T> out, int delta) {

            // indices
            dim3_t dims = in.dims();
            int3 idx = Index3D();
            if (idx < dims) {
                int3 idx2 = idx;
                idx2.z = (idx.z + delta + dims.z) % dims.z;
                out[idx] = in[idx2];
            }
        }

        template <typename T>
        DeviceArray<T> roll(const DeviceArray<T> &arr, int delta) {

            auto dims = arr.dims();
            DeviceArray<T> out(dims);
            Grid grid(dims);
            roll_kernel<T> <<< grid.blocks(), grid.threads()>>>(arr, out, delta);
            return out;
        }
        // explicit instantiation
        template DeviceArray<float> roll(const DeviceArray<float> &, int);
        template DeviceArray<double> roll(const DeviceArray<double> &, int);
        template DeviceArray<complex_t<float>> roll(
            const DeviceArray<complex_t<float>> &, int);
        template DeviceArray<complex_t<double>> roll(
            const DeviceArray<complex_t<double>> &, int);

        /* --------------------------------------------------------------------
         */
        // fftshift
        template <typename T>
        __global__ void phase_shift_kernel(const DeviceMemory<complex_t<T>> in,
            DeviceMemory<complex_t<T>> out, T delta) {

            dim3_t dims = in.dims();
            int3 idx = Index3D();
            if (idx < dims) {
                T q = 2 * M_PI * idx.z / dims.z * delta;
                out[idx] = complex_t<T>(cuda::std::cos(q), cuda::std::sin(q)) *
                    in[idx];
            }
        }

        template <typename T>
        DeviceArray<complex_t<T>> phase_shift(
            const DeviceArray<complex_t<T>> &in, T delta) {

            DeviceArray<complex_t<T>> out(in.dims());
            Grid grid(in.dims());
            phase_shift_kernel<T>
            <<< grid.blocks(), grid.threads()>>>(in, out, delta);
            return out;
        }

        // explicit instantiation
        template DeviceArray<complex_t<float>> phase_shift(
            const DeviceArray<complex_t<float>> &, float);
        template DeviceArray<complex_t<double>> phase_shift(
            const DeviceArray<complex_t<double>> &, double);

    } // namespace gpu
} // namespace tomocam
