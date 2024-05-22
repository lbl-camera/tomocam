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

#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#include "types.h"
#include "dev_array.h"
#include "gpu/dev_memory.cuh"
#include "gpu/utils.cuh"

namespace tomocam {
    namespace gpu {

        template <typename T>
        __global__ void pad_left_kernel(DeviceMemory<T> in, DeviceMemory<T> out, int shift) {

            int3 idx = Index3D();
            if (idx < out.dims()) {
                int3 out_idx = {idx.x, idx.y, idx.z + shift};
                out[out_idx] = in[idx];
            }
        }

        template <typename T>
        DeviceArray<T> pad_left(DeviceArray<T> in, int padding, cudaStream_t s) {

            int shift = std::abs(padding);
            int ncols = in.ncols() + 2 * std::abs(padding) - 1;
            dim3_t new_dim = {in.nslices(), in.nrows(), ncols};
            DeviceArray<T> out(new_dim);
            SAFE_CALL(cudaMemset(out.dev_ptr(), 0, out.size() * sizeof(T)));
        
            // cuda kernel launch
            Grid grid(in.dims());
            pad_left_kernel<T><<<grid.blocks(), grid.threads(), 0, s>>>(in, out, shift);
            return out;
        }


        template <typename T>
        __global__ void pad_right_kernel(DeviceMemory<T> in, DeviceMemory<T> out) {

            int3 idx = Index3D();
            if (idx < out.dims()) 
                out[idx] = in[idx];
        }

        template <typename T>
        DeviceArray<T> pad_right(DeviceArray<T> in, int padding, cudaStream_t s) {

            int ncols = in.ncols() + 2 * std::abs(padding) - 1;
            dim3_t new_dim = {in.nslices(), in.nrows(), ncols};
            DeviceArray<T> out(new_dim);
        
            // cuda kernel launch
            Grid grid(in.dims());
            pad_right_kernel<T><<<grid.blocks(), grid.threads(), 0, s>>>(in, out);
            return out;
        }

        template <typename T>
        DeviceArray<T> pad1d(const DeviceArray<T> & in, int padding, cudaStream_t s) {

            if (padding < 0)
                return pad_left(in, padding, s);
            else
                return pad_right(in, padding, s);
        }
        // specializations
        template DeviceArray<float> pad1d(const DeviceArray<float> &, int, cudaStream_t);
        template DeviceArray<double> pad1d(const DeviceArray<double> &, int, cudaStream_t);


        /* two-dimensional padding */
        template<typename T>
        __global__ void pad2d_kernel(DeviceMemory<T> in, DeviceMemory<T> out, int shift) {

            int3 idx = Index3D();
            if (idx < out.dims()) {
                int3 out_idx = {idx.x, idx.y + shift, idx.z + shift};
                out[out_idx] = in[idx];
            }
        }

        template <typename T>
        DeviceArray<T> pad2d(const DeviceArray<T> &in, int padding, cudaStream_t s) {

            int shift = std::abs(padding);
            int ncols = in.ncols() + 2 * std::abs(padding);
            int nrows = in.nrows() + 2 * std::abs(padding);
            dim3_t new_dim = {in.nslices(), nrows, ncols};
            DeviceArray<T> out(new_dim);
        
            // cuda kernel launch
            Grid grid(out.dims());
            pad2d_kernel<T><<<grid.blocks(), grid.threads(), 0, s>>>(in, out, shift);
            return out;
        }
        // specializations
        template DeviceArray<float> pad2d(const DeviceArray<float> &, int, cudaStream_t);
        template DeviceArray<double> pad2d(const DeviceArray<double> &, int, cudaStream_t);



        /* two-dimensional crop */
        template<typename T>
        __global__ void crop_kernel(DeviceMemory<T> in, DeviceMemory<T> out, int shift) {

            int3 idx = Index3D();
            if (idx < out.dims()) {
                int3 in_idx = {idx.x, idx.y + shift, idx.z + shift};
                out[idx] = in[in_idx];
            }
        }

        template <typename T>
        DeviceArray<T> unpad2d(const DeviceArray<T> &in, int padding, cudaStream_t s) {

            int shift = std::abs(padding);
            int ncols = in.ncols() - 2 * std::abs(padding);
            dim3_t new_dim = {in.nslices(), in.nrows(), ncols};
            DeviceArray<T> out(new_dim);
        
            // cuda kernel launch
            Grid grid(out.dims());
            crop_kernel<T><<<grid.blocks(), grid.threads(), 0, s>>>(in, out, shift);
            return out;
        }

        // specializations
        template DeviceArray<float> unpad2d(const DeviceArray<float> &, int, cudaStream_t);
        template DeviceArray<gpu::complex_t<float>> unpad2d(
            const DeviceArray<gpu::complex_t<float>> &, int, cudaStream_t);
        template DeviceArray<double> unpad2d(const DeviceArray<double> &, int, cudaStream_t);
        template DeviceArray<gpu::complex_t<double>> unpad2d(
            const DeviceArray<gpu::complex_t<double>> &, int, cudaStream_t);

    } // namespace gpu
} // namespace tomocam
