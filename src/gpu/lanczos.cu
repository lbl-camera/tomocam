
#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"

#include "gpu/dev_memory.cuh"
#include "utils.cuh"

namespace tomocam::gpu {
    const int LANCZOS_A = 2;

    template <typename T>
    __device__ T lanczos_window(T x) {
        if (x == 0) { return 1; }
        if (abs(x) >= LANCZOS_A) { return 0; }
        return LANCZOS_A * sin(x * M_PI) * sin(x * M_PI / LANCZOS_A) /
               (x * x * M_PI * M_PI);
    }

    template <typename T>
    __global__ void lanczos_upsampling(const DeviceMemory<T> in,
        DeviceMemory<T> out) {

        int3 idx = Index3D();
        auto dims = in.dims();
        auto d2 = out.dims();

        T scalex = (T)d2.x / (T)dims.x;
        T scaley = (T)d2.y / (T)dims.y;
        T scalez = (T)d2.z / (T)dims.z;
        if (idx < d2) {
            T cenx = idx.x / scalex;
            T ceny = idx.y / scaley;
            T cenz = idx.z / scalez;

            // Calculate the bounds for the Lanczos kernel
            int ibeg = max(0, (int)floor(cenx) - LANCZOS_A + 1);
            int jbeg = max(0, (int)floor(ceny) - LANCZOS_A + 1);
            int kbeg = max(0, (int)floor(cenz) - LANCZOS_A + 1);
            int iend = min(dims.x - 1, (int)floor(cenx) + LANCZOS_A);
            int jend = min(dims.y - 1, (int)floor(ceny) + LANCZOS_A);
            int kend = min(dims.z - 1, (int)floor(cenz) + LANCZOS_A);

            T sum = 0;
            T sumw = 0;
            for (int i = ibeg; i <= iend; i++) {
                for (int j = jbeg; j <= jend; j++) {
                    for (int k = kbeg; k <= kend; k++) {
                        T w = lanczos_window(cenx - i) *
                              lanczos_window(ceny - j) *
                              lanczos_window(cenz - k);
                        sum += w * in(i, j, k);
                        sumw += w;
                    }
                }
            }
            out[idx] = sum / sumw;
        }
    }

    template <typename T>
    DeviceArray<T> lanczos_upsampling(const DeviceArray<T> &in, dim3_t dims) {

        DeviceArray<T> out(dims);
        Grid grid(out.dims());
        lanczos_upsampling<T><<<grid.blocks(), grid.threads()>>>(in, out);
        return out;
    }

    // explicit instantiation
    template DeviceArray<float> lanczos_upsampling(const DeviceArray<float> &,
        dim3_t);
    template DeviceArray<double> lanczos_upsampling(const DeviceArray<double> &,
        dim3_t);

} // namespace tomocam::gpu
