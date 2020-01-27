
#include "dev_array.h"

namespace tomocam {
    __global__ void kaiser(float *window, float W, float beta, size_t len) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < len) {
            float r   = 0.5 * (W - 1) / (len - 1) * i;
            float x   = cyl_bessel_i0(beta * sqrtf(1.f - powf(2 * r / W, 2.f)));
            float y   = cyl_bessel_i0(beta);
            window[i] = x / y;
        }
    }

    void kaiser_window(kernel_t &kernel, float width, float beta, size_t len, int device) {
        cudaSetDevice(device);

        kernel.set_size(len);
        kernel.set_device_id(device);
        kernel.set_radius(0.5 * (width - 1));
        kernel.set_beta(beta);

        dim3 threads(256);
        dim3 tblocks(len / threads.x + 1);

        float *d_window = NULL;
        cudaMalloc((void **) &d_window, len * sizeof(float));
        kaiser <<< tblocks, threads >>> (d_window, width, beta, len);
        kernel.set_d_array(d_window);
    }

} // namespace tomocam
