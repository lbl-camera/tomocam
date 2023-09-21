
#include "dev_array.h"
#include "utils.cuh"

namespace tomocam {

    template <typename T>
    __global__ void 
    upsample_kernel(DeviceArray<T> inp, DeviceArray<T> out) {

        // global indices
        auto z = blockDim.x * blockIdx.x + threadIdx.x;
        auto y = blockDim.y * blockIdx.y + threadIdx.y;
        auto x = blockDim.z * blockIdx.z + threadIdx.z;

        auto dims = inp.dims();
        // this slice
        if ((x < dims.x) && (y < dims.y) && (z < dims.z)) {
            out(2*x, 2*y, 2*z)     = inp(x, y, z);
            out(2*x, 2*y, 2*z+1)   = inp(x, y, z);
            out(2*x, 2*y+1, 2*z)   = inp(x, y, z);
            out(2*x, 2*y+1, 2*z+1) = inp(x, y, z);

            out(2*x+1, 2*y, 2*z)     = inp(x, y, z);
            out(2*x+1, 2*y, 2*z+1)   = inp(x, y, z);
            out(2*x+1, 2*y+1, 2*z)   = inp(x, y, z);
            out(2*x+1, 2*y+1, 2*z+1) = inp(x, y, z);
        }
    } // upsample_kernel

    template <typename T>
    void upsample(DeviceArray<T> &inp, DeviceArray<T> &out) {
        Grid grid(inp.dims()); 
        upsample_kernel<<<grid.blocks(), grid.threads()>>>(inp, out);
    }
    template void upsample(DeviceArray<float> &, DeviceArray<float> &);


    template <typename T>
    __global__ void 
    downsample_kernel(DeviceArray<T> inp, DeviceArray<T> out, int n) {

        auto idx = Index3D();
        int3 idx0 = {n*idx.x, n*idx.y, n*idx.z};  
        if (idx < out.dims()) {
            out[idx] = inp[idx0];
        }
    }

    template <typename T>
    void downsample(DeviceArray<T> &inp, DeviceArray<T> &out, int n) {
        Grid grid(out.dims()); 
        downsample_kernel<<<grid.blocks(), grid.threads()>>>(inp, out, n);
    }
    template void downsample(DeviceArray<float> &, DeviceArray<float> &, int n);

} // namespace
