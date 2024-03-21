#include <cuda.h>
#include <cuda/std/complex>

#include "common.h"

namespace tomocam {
    namespace gpu_ops {
        template <typename T>
        __global__ void circshift_kernel(T *out, T *in, int nrows, int ncols, int yshift, int xshift) {

            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int j = blockIdx.y * blockIdx.y + threadIdx.y;
            if ((i < nrows) && (j < ncols)) {
                int ii = (i + yshift) % nrows;
                int jj = (j + xshift) % ncols;
                out[ii * ncols + jj] = in[i * ncols + j];
            }
        }

        template <typename T>
        void circshift(T *out, T *in, int nrows, int ncols, int yshift, int xshift) {

            dim3 threads(32, 16, 1);
            int n1 = ceili(nrows, threads.x);
            int n2 = ceili(ncols, threads.y);
            dim3 blocks(n1, n2, 1);
            circshift_kernel<<<blocks, threads>>>(out, in, nrows, ncols, yshift, xshift);
        }

        template void circshift<float>(float *, float *, int, int, int, int);
        template void circshift<double>(double *, double *, int, int, int, int);
        template void circshift<cuda::std::complex<float>>(cuda::std::complex<float> *, cuda::std::complex<float> *, int, int, int, int); 
        template void circshift<cuda::std::complex<double>>(cuda::std::complex<double> *, cuda::std::complex<double> *, int, int, int, int);

    } // namespace gpu_ops
} // namespace tomocam
