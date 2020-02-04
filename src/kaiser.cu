
#include "dev_array.h"
#include "kernel.h"

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

    kernel_t kaiser_window(float width, float beta, size_t len) {

        dim3 threads(256);
        dim3 tblocks(len / threads.x + 1);

        float *d_window = NULL;
        cudaMalloc((void **) &d_window, len * sizeof(float));

        kaiser <<< tblocks, threads >>> (d_window, width, beta, len);
        kernel_t kernel;
        kernel.set_radius((width-1)/2);
        kernel.set_beta(beta);
        kernel.set_dims(dim3_t(1, 1, len));
        kernel.set_d_array(d_window);
        return kernel;
    }
} // namespace tomocam
