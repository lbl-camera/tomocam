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

#include "dev_array.h"
#include "gpu/dev_memory.cuh"
#include "gpu/padding.cuh"
#include "gpu/utils.cuh"

namespace tomocam::gpu {

    template <typename T>
    __global__ void pad1d_kernel(DeviceMemory<T> in, DeviceMemory<T> out,
        int shift) {

        int3 idx = Index3D();
        if (idx < in.dims()) {
            int3 out_idx = {idx.x, idx.y, idx.z + shift};
            out[out_idx] = in[idx];
        }
    }

    template <typename T>
    DeviceArray<T> pad1d(const DeviceArray<T> &in, int padding, PadType type) {

        // allocate the output array
        int ncols = in.ncols() + std::abs(padding);
        dim3_t new_dim = {in.nslices(), in.nrows(), ncols};
        DeviceArray<T> out(new_dim);
        SAFE_CALL(cudaMemset(out.dev_ptr(), 0, out.bytes()));

        // calculate the shift
        int shift = std::abs(padding) / 2;
        if (type == PadType::RIGHT) shift = 0;
        if (type == PadType::LEFT) shift = shift;

        // cuda kernel launch
        Grid grid(in.dims());
        pad1d_kernel<T><<<grid.blocks(), grid.threads()>>>(in, out, shift);
        return out;
    }

    // specializations
    template DeviceArray<float> pad1d(const DeviceArray<float> &, int, PadType);
    template DeviceArray<double> pad1d(const DeviceArray<double> &, int,
        PadType);

    /* one-dimensional crop */
    template <typename T>
    __global__ void crop_kernel(DeviceMemory<T> in, DeviceMemory<T> out,
        int shift) {

        int3 idx = Index3D();
        if (idx < out.dims()) {
            int3 in_idx = {idx.x, idx.y, idx.z + shift};
            out[idx] = in[in_idx];
        }
    }

    template <typename T>
    DeviceArray<T> unpad1d(const DeviceArray<T> &in, int padding,
        PadType type) {

        // new dimensions
        dim3_t new_dim = {in.nslices(), in.nrows(),
            in.ncols() - std::abs(padding)};
        DeviceArray<T> out(new_dim);

        // set the shift depending on the padding type
        int shift = std::abs(padding) / 2;
        if (type == PadType::RIGHT) shift = 0;
        if (type == PadType::LEFT) shift = shift;

        // cuda kernel launch
        Grid grid(out.dims());
        crop_kernel<T><<<grid.blocks(), grid.threads()>>>(in, out, shift);
        return out;
    }
    // specializations
    template DeviceArray<float> unpad1d(const DeviceArray<float> &, int,
        PadType);
    template DeviceArray<double> unpad1d(const DeviceArray<double> &, int,
        PadType);
    template DeviceArray<complex_t<float>> unpad1d(
        const DeviceArray<complex_t<float>> &, int, PadType);
    template DeviceArray<complex_t<double>> unpad1d(
        const DeviceArray<complex_t<double>> &, int, PadType);

    /* two-dimensional padding */
    template <typename T>
    __global__ void pad2d_kernel(DeviceMemory<T> in, DeviceMemory<T> out,
        int shift) {

        int3 idx = Index3D();
        if (idx < in.dims()) {
            int3 out_idx = {idx.x, idx.y + shift, idx.z + shift};
            out[out_idx] = in[idx];
        }
    }

    template <typename T>
    DeviceArray<T> pad2d(const DeviceArray<T> &in, int padding, PadType type) {

        // allocate the output array
        int ncols = in.ncols() + std::abs(padding);
        int nrows = in.nrows() + std::abs(padding);
        dim3_t new_dim = {in.nslices(), nrows, ncols};
        DeviceArray<T> out(new_dim);
        SAFE_CALL(cudaMemset(out.dev_ptr(), 0, out.bytes()));

        // calculate the shift
        int shift = std::abs(padding) / 2;
        if (type == PadType::RIGHT) shift = 0;
        if (type == PadType::LEFT) shift = std::abs(padding);

        // cuda kernel launch
        Grid grid(in.dims());
        pad2d_kernel<T><<<grid.blocks(), grid.threads()>>>(in, out, shift);
        return out;
    }
    // specializations
    template DeviceArray<float> pad2d(const DeviceArray<float> &, int, PadType);
    template DeviceArray<double> pad2d(const DeviceArray<double> &, int,
        PadType);

    /* two-dimensional crop */
    template <typename T>
    __global__ void crop2d_kernel(DeviceMemory<T> in, DeviceMemory<T> out,
        int shift) {

        int3 idx = Index3D();
        if (idx < out.dims()) {
            int3 in_idx = {idx.x, idx.y + shift, idx.z + shift};
            out[idx] = in[in_idx];
        }
    }

    template <typename T>
    DeviceArray<T> unpad2d(const DeviceArray<T> &in, int padding,
        PadType type) {

        int nrows = in.nrows() - std::abs(padding);
        int ncols = in.ncols() - std::abs(padding);
        dim3_t new_dim = {in.nslices(), nrows, ncols};
        DeviceArray<T> out(new_dim);

        // shift for symmetric padding
        int shift = std::abs(padding) / 2;
        if (type == PadType::RIGHT) shift = 0;
        if (type == PadType::LEFT) shift = std::abs(padding);

        // cuda kernel launch
        Grid grid(new_dim);
        crop2d_kernel<T><<<grid.blocks(), grid.threads()>>>(in, out, shift);
        return out;
    }
    // specializations for float and double
    template DeviceArray<float> unpad2d(const DeviceArray<float> &, int,
        PadType);
    template DeviceArray<double> unpad2d(const DeviceArray<double> &, int,
        PadType);

    template DeviceArray<complex_t<float>> unpad2d(
        const DeviceArray<complex_t<float>> &, int, PadType);
    template DeviceArray<complex_t<double>> unpad2d(
        const DeviceArray<complex_t<double>> &, int, PadType);
} // namespace tomocam::gpu
