
#include "dev_array.h"
#include "utils.cuh"

namespace tomocam {

    __global__ void 
    upsample_kernel(DeviceMemory<float> inp, DeviceMemory<float> out) {

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

    void upsample(DeviceArray<float> &inp, DeviceArray<float> &out) {
        Grid grid(inp.dims()); 
        upsample_kernel<<<grid.blocks(), grid.threads()>>>(inp, out);
    }


    __global__ void downsample_kernel(DeviceMemory<float> inp, DeviceMemory<float> out, int n) {

        auto idx = Index3D();
        int3 idx0 = {n*idx.x, n*idx.y, n*idx.z};  
        if (idx < out.dims()) {
            out[idx] = inp[idx0];
        }
    }

    void downsample(DeviceArray<float> &inp, DeviceArray<float> &out, int n) {
        Grid grid(out.dims()); 
        downsample_kernel<<<grid.blocks(), grid.threads()>>>(inp, out, n);
    }

} // namespace
