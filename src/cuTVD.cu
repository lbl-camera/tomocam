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

#include "polarsample.h"
#include <cuda.h>

const int DIMX    = 16;
const int DIMY    = 4;
const int DIMZ    = 4;
const int WORK    = 8;
const int sNX     = DIMX + 2;
const int sNY     = DIMY + 2;
const int sNZ     = DIMZ + 2;
const float MRF_C = .001;
const float MRF_Q = 2;
__constant__ int3 dims;

__device__ const float FILTER[3][3][3] = {{{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}},
    {{0.037, 0.0523, 0.037}, {0.0532, 0., 0.0523}, {0.037, 0.0523, 0.037}},
    {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}}};

__device__ float wght(int i, int j, int k) { return FILTER[i][j][k]; }

__device__ int globalIdx(int x, int y, int z) { return (z * dims.y * dims.z + dims.z * y + x); }

__device__ float pot_func(float delta, float MRF_P, float MRF_SIGMA) {
    return ((pow(fabs(delta) / MRF_SIGMA, MRF_Q)) / (MRF_C + pow(fabs(delta) / MRF_SIGMA, MRF_Q - MRF_P)));
}

__device__ float deriv_potFCN(float delta, float MRF_P, float MRF_SIGMA) {
    float MRF_SIGMA_Q   = pow(MRF_SIGMA, MRF_Q);
    float MRF_SIGMA_Q_P = pow(MRF_SIGMA, MRF_Q - MRF_P);

    float temp1 = pow(fabs(delta), MRF_Q - MRF_P) / MRF_SIGMA_Q_P;
    float temp2 = pow(fabs(delta), MRF_Q - 1);
    float temp3 = MRF_C + temp1;

    if (delta < 0.0) {
        return ((-1 * temp2 / (temp3 * MRF_SIGMA_Q)) * (MRF_Q - ((MRF_Q - MRF_P) * temp1) / (temp3)));
    } else if (delta > 0.0) {
        return ((temp2 / (temp3 * MRF_SIGMA_Q)) * (MRF_Q - ((MRF_Q - MRF_P) * temp1) / (temp3)));
    } else {
        return 0; // MRF_Q / (MRF_SIGMA_Q*MRF_C);
    }
}

/*Second Derivative of the potential function at zero */
__device__ float second_deriv_potFunc_zero(float MRF_SIGMA) {
    float MRF_SIGMA_Q = pow(MRF_SIGMA, MRF_Q);
    return MRF_Q / (MRF_SIGMA_Q * MRF_C);
}

__global__ void tvd_update_kernel(float mrf_p, float mrf_sigma, float *val, float *tvd) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z;
    int x = i + blockDim.x * blockIdx.x;
    int y = j + blockDim.y * blockIdx.y;
    int z = k + blockDim.z * blockIdx.z;

    // last thread in the block
    int in = min(nx - blockIdx.x * blockDim.x - 1, blockDim.x - 1);
    int jn = min(ny - blockIdx.y * blockDim.y - 1, blockDim.y - 1);
    int kn = min(nz - blockIdx.z * blockDim.z - 1, blockDim.z - 1);

    if ((x < nx) && (y < ny) && (z < nz)) {

        int gid = globalIdx(x, y, z);

        /* copy values into shared memory.
         * Max size of shared memory = 64 x 1024 Bytes
         * which translates to  8192 complex number
         */
        __shared__ float s_val[sNZ][sNY][sNX];

        // copy from global memory
        s_val[k + 1][j + 1][i + 1] = val[gid];

        /* copy ghost cells, on all 6 faces */
        // x = 0 face
        if (i == 0) {
            if (x > 0) s_val[k + 1][j + 1][i] = val[globalIdx(x - 1, y, z)];
            else
                s_val[k + 1][j + 1][i] = 0.f;
        }

        // x = Nx-1 face
        if (i == in) {
            if (x < nx - 1) s_val[k + 1][j + 1][i + 2] = val[globalIdx(x + 1, y, z)];
            else
                s_val[k + 1][j + 1][i + 2] = 0.f;
        }
        __syncthreads();

        if (j == 0) {
            if (y > 0) s_val[k + 1][j][i + 1] = val[globalIdx(x, y - 1, z)];
            else
                s_val[k + 1][j][i + 1] = 0.f;
        }

        if (j == jn) {
            if (y < ny - 1) s_val[k + 1][j + 2][i + 1] = val[globalIdx(x, y + 1, z)];
            else
                s_val[k + 1][j + 2][i + 1] = 0.f;
        }
        __syncthreads();

        if (k == 0) {
            if (z > 0) s_val[k][j + 1][i + 1] = val[globalIdx(x, y, z - 1)];
            else
                s_val[k][j + 1][i + 1] = 0.f;
        }

        if (k == kn) {
            if (z < nz - 1) s_val[k + 2][j + 1][i + 1] = val[globalIdx(x, y, z + 1)];
            else
                s_val[k + 2][j + 1][i + 1] = 0.f;
        }
        __syncthreads();

        /* copy ghost cells along 12 edges  */
        // copy ghost-cells along x-direction
        if (j == 0) {
            if (k == 0) {
                if ((y > 0) && (z > 0)) s_val[k][j][i + 1] = val[globalIdx(x, y - 1, z - 1)];
                else
                    s_val[k][j][i + 1] = 0.f;
            }
            if (k == kn) {
                if ((y > 0) && (z < nz - 1)) s_val[k + 2][j][i + 1] = val[globalIdx(x, y - 1, z + 1)];
                else
                    s_val[k + 2][j][i + 1] = 0.f;
            }
        }
        if (j == jn) {
            if (k == 0) {
                if ((y < ny - 1) && (z > 0)) s_val[k][j + 2][i + 1] = val[globalIdx(x, y + 1, z - 1)];
                else
                    s_val[k][j + 2][i + 1] = 0.f;
            }
            if (k == kn) {
                if ((y < ny - 1) && (z < nz - 1)) s_val[k + 2][j + 2][i + 1] = val[globalIdx(x, y + 1, z + 1)];
                else
                    s_val[k + 2][j + 2][i + 1] = 0.f;
            }
        }
        __syncthreads();

        // copy ghost-cells along y-direction
        if (k == 0) {
            if (i == 0) {
                if ((x > 0) && (z > 0)) s_val[k][j + 1][i] = val[globalIdx(x - 1, y, z - 1)];
                else
                    s_val[k][j + 1][i] = 0.f;
            }
            if (i == in) {
                if ((x < nx - 1) && (z > 0)) s_val[k][j + 1][i + 2] = val[globalIdx(x + 1, y, z - 1)];
                else
                    s_val[k][j + 1][i + 2] = 0.f;
            }
        }
        if (k == kn) {
            if (i == 0) {
                if ((x > 0) && (z < nz - 1)) s_val[k + 2][j + 1][i] = val[globalIdx(x - 1, y, z + 1)];
                else
                    s_val[k + 2][j + 1][i] = 0.f;
            }
            if (i == in) {
                if ((x < nx - 1) && (z < nz - 1)) s_val[k + 2][j + 1][i + 2] = val[globalIdx(x + 1, y, z + 1)];
                else
                    s_val[k + 2][j + 1][i + 2] = 0.f;
            }
        }
        __syncthreads();

        if (i == 0) {
            if (j == 0) {
                if ((x > 0) && (y > 0)) s_val[k + 1][j][i] = val[globalIdx(x - 1, y - 1, z)];
                else
                    s_val[k + 1][j][i] = 0.f;
            }
            if (j == jn) {
                if ((x > 0) && (y < ny - 1)) s_val[k + 1][j + 2][i] = val[globalIdx(x - 1, y + 1, z)];
                else
                    s_val[k + 1][j + 2][i] = 0.f;
            }
        }
        if (i == in) {
            if (j == 0) {
                if ((x < nx - 1) && (y > 0)) s_val[k + 1][j][i + 2] = val[globalIdx(x + 1, y - 1, z)];
                else
                    s_val[k + 1][j][i + 2] = 0.f;
            }
            if (j == jn) {
                if ((x < nx - 1) && (y < ny - 1)) s_val[k + 1][j + 2][i + 2] = val[globalIdx(x + 1, y + 1, z)];
                else
                    s_val[k + 1][j + 2][i + 2] = 0.f;
            }
        }
        __syncthreads();

        /*  copy  ghost cells along 16 corners */
        if (k == 0) {
            if (j == 0) {
                if (i == 0) {
                    if ((x > 0) && (y > 0) && (z > 0)) s_val[k][j][i] = val[globalIdx(x - 1, y - 1, z - 1)];
                    else
                        s_val[k][j][i] = 0.f;
                }
                if (i == in) {
                    if ((x < nx - 1) && (y > 0) && (z > 0)) s_val[k][j][i + 2] = val[globalIdx(x + 1, y - 1, z - 1)];
                    else
                        s_val[k][j][i + 2] = 0.f;
                }
            }
            if (j == jn) {
                if (i == 0) {
                    if ((x > 0) && (y < ny - 1) && (z > 0)) s_val[k][j + 2][i] = val[globalIdx(x - 1, y + 1, z - 1)];
                    else
                        s_val[k][j + 2][i] = 0.f;
                }
                if (i == in) {
                    if ((x < nx - 1) && (y < ny - 1) && (z > 0))
                        s_val[k][j + 2][i + 2] = val[globalIdx(x + 1, y + 1, z - 1)];
                    else
                        s_val[k][j + 2][i + 2] = 0.f;
                }
            }
        }
        if (k == kn) {
            if (j == 0) {
                if (i == 0) {
                    if ((x > 0) && (y > 0) && (z < nz - 1)) s_val[k + 2][j][i] = val[globalIdx(x - 1, y - 1, z + 1)];
                    else
                        s_val[k + 2][j][i] = 0.f;
                }
                if (i == in) {
                    if ((x < nx - 1) && (y > 0) && (z < nz - 1))
                        s_val[k + 2][j][i + 2] = val[globalIdx(x + 1, y - 1, z + 1)];
                    else
                        s_val[k + 2][j][i + 2] = 0.f;
                }
            }
            if (j == jn) {
                if (i == 0) {
                    if ((x > 0) && (y < ny - 1) && (z < nz - 1))
                        s_val[k + 2][j + 2][i] = val[globalIdx(x - 1, y + 1, z + 1)];
                    else
                        s_val[k + 2][j + 2][i] = 0.f;
                }
                if (i == in) {
                    if ((x < nx - 1) && (y < ny - 1) && (z < nz - 1))
                        s_val[k + 2][j + 2][i + 2] = val[globalIdx(x + 1, y + 1, z + 1)];
                    else
                        s_val[k + 2][j + 2][i + 2] = 0.f;
                }
            }
        }
        __syncthreads();

        float v    = s_val[k + 1][j + 1][i + 1];
        float temp = 0.f;
        for (int iz = 0; iz < 3; iz++)
            for (int iy = 0; iy < 3; iy++)
                for (int ix = 0; ix < 3; ix++)
                    temp += wght(iz, iy, ix) * deriv_potFCN(v - s_val[k + iz][j + iy][i + ix], mrf_p, mrf_sigma);
        tvd[gid] += temp;
    }
}

__global__ void hessian_zero_kernel(int nc, int nr, int ns, float mrf_sigma, float *val, float *hessian) {

    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z;
    int x = i + blockDim.x * blockIdx.x;
    int y = j + blockDim.y * blockIdx.y;
    int z = k + blockDim.z * blockIdx.z;

    if ((x < nc) && (y < nr) && (z < ns)) {
        int gid = nc * nr * z + nc * y + x;
        for (int iz = 0; iz < 3; iz++)
            for (int iy = 0; iy < 3; iy++)
                for (int ix = 0; ix < 3; ix++) temp += wght(iz, iy, ix) * second_deriv_potFunc_zero(mrf_sigma);
        hessian[gid] += temp;
    }
}

void addTVD(int3 h_dims, float mrf_p, float mrf_sigma, float *objfn, float *val, cudaStream_t stream) {

    int nslice = h_dims.x;
    int nrow   = h_dims.y;
    int ncol   = h_dims.z;
    int GRIDX  = ncol % DIMX > 0 ? ncol / DIMX + 1 : ncol / DIMX;
    int GRIDY  = nrow % DIMY > 0 ? nrow / DIMY + 1 : nrow / DIMY;
    int GRIDZ  = nslice % DIMZ > 0 ? nslice / DIMZ + 1 : nslice / DIMZ;

    // block dims
    dim3 block(DIMX, DIMY, DIMZ);
    dim3 grid(GRIDX, GRIDY, GRIDZ);

    /* copy the grid dimensions to constant memeory */
    cudaError_t status;
    status = cudaMemcpyToSymbol(dims, &h_dims, sizeof(int3));

    tvd_update_kernel<<<grid, block, 0, stream>>>(mrf_p, mrf_sigma, val, objfn);
}

void calcHessian(int nslice, int nrow, int ncol, float mrf_sigma, float *volume, float *hessian, cudaStream_t stream) {

    int GRIDX = ncol % DIMX > 0 ? ncol / DIMX + 1 : ncol / DIMX;
    int GRIDY = nrow % DIMY > 0 ? nrow / DIMY + 1 : nrow / DIMY;
    int GRIDZ = nslice % DIMZ > 0 ? nslice / DIMZ + 1 : nslice / DIMZ;

    // block dims
    dim3 block(DIMX, DIMY, DIMZ);
    dim3 grid(GRIDX, GRIDY, GRIDZ);

    // update hessain inplace
    hessian_zero_kernel<<<grid, block, 0, stream>>>(ncol, nrow, nslice, mrf_sigma, volume, hessian);
}
