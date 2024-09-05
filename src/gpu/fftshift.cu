
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
                idx2.z = (idx.z + delta) % dims.z;
                out[idx] = in[idx2];
            }
        }

        template <typename T>
        DeviceArray<T> roll(const DeviceArray<T> &arr, int delta) {

            auto dims = arr.dims();
            DeviceArray<T> out(dims);
            Grid grid(dims);
            roll_kernel<T><<<grid.blocks(), grid.threads()>>>(arr, out, delta);
            return out;
        }
        // explicit instantiation
        template DeviceArray<float> roll(const DeviceArray<float> &, int);
        template DeviceArray<double> roll(const DeviceArray<double> &, int);
        template DeviceArray<gpu::complex_t<float>> roll(
            const DeviceArray<gpu::complex_t<float>> &, int);
        template DeviceArray<gpu::complex_t<double>> roll(
            const DeviceArray<gpu::complex_t<double>> &, int);

    } // namespace gpu
} // namespace tomocam
