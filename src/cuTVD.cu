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

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "utils.cuh"

namespace tomocam {
    const int NX = 1;
    const int NY = 16;
    const int NZ = 16;
    const float MRF_Q = 2.f;
    const float MRF_C = 0.001f;

    __device__ const float FILTER[3][3][3] = {
        {{0.0302, 0.0370, 0.0302}, {0.0370, 0.0523, 0.0370}, {0.0302, 0.0370, 0.0302}},
        {{0.0370, 0.0523, 0.0370}, {0.0532, 0.0000, 0.0523}, {0.0370, 0.0523, 0.0370}},
        {{0.0302, 0.0370, 0.0302}, {0.0370, 0.0523, 0.0370}, {0.0302, 0.0370, 0.0302}}};

    __forceinline__ __device__ float weight(int i, int j, int k) { return FILTER[i][j][k]; }

    /*
     *            (|d| / sigma)^q
     *  f(d) =  -------------------
     *          c + (|d| / sigma)^(q-p)
     */
    __device__ float pot_func(float delta, float MRF_P, float MRF_SIGMA) {
        return ((powf(fabs(delta) / MRF_SIGMA, MRF_Q)) / (MRF_C + powf(fabs(delta) / MRF_SIGMA, MRF_Q - MRF_P)));
    }

    __device__ float deriv_potFCN(float delta, float MRF_P, float MRF_SIGMA) {
        float MRF_SIGMA_Q = powf(MRF_SIGMA, MRF_Q);
        float MRF_SIGMA_Q_P = powf(MRF_SIGMA, MRF_Q - MRF_P);

        float temp1 = powf(fabs(delta), MRF_Q - MRF_P) / MRF_SIGMA_Q_P;
        float temp2 = powf(fabs(delta), MRF_Q - 1);
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
        float MRF_SIGMA_Q = powf(MRF_SIGMA, MRF_Q);
        return MRF_Q / (MRF_SIGMA_Q * MRF_C);
    }

    __global__ void tvd_update_kernel(dev_arrayf model, dev_arrayf objfn, float p, float sigma) {

        // thread ids
        int i = threadIdx.x;
        int j = threadIdx.y;
        int k = threadIdx.z;

        // global offsets
        int I0 = blockDim.x * blockIdx.x;
        int J0 = blockDim.y * blockIdx.y;
        int K0 = blockDim.z * blockIdx.z;

        // global ids
        int x = I0 + i;
        int y = J0 + j;
        int z = K0 + k;

        // last thread in the block
        dim3_t dims = objfn.dims();
        int imax = min(dims.x - I0 - 1, blockDim.x - 1);
        int jmax = min(dims.y - J0 - 1, blockDim.y - 1);
        int kmax = min(dims.z - K0 - 1, blockDim.z - 1);

        if ((x < dims.x) && (y < dims.y) && (z < dims.z)) {

            // size of the array
            dim3_t dims = objfn.dims();

            /* copy values into shared memory. */
            __shared__ float s_val[NX + 2][NY + 2][NZ + 2];

            // copy from global memory
            s_val[i + 1][j + 1][k + 1] = model.at(x, y, z);

            __syncthreads();
            /* copy ghost cells, on all 6 faces */
            // x = 0 face
            if (i == 0) 
                s_val[i][j + 1][k + 1] = model.at(x - 1, y, z);

            // x = Nx-1 face
            if (i == imax)
                s_val[i + 2][j + 1][k + 1] = model.at(x + 1, y, z);
            __syncthreads();

            if (j == 0)
                s_val[i + 1][j][k + 1] = model.at(x, y - 1, z);

            if (j == jmax)
                s_val[i + 1][j + 2][k + 1] = model.at(x, y + 1, z);
            __syncthreads();

            if (k == 0)
                s_val[i + 1][j + 1][k] = model.at(x, y, z - 1);

            if (k == kmax)
                s_val[i + 1][j + 1][k + 2] = model.at(x, y, z + 1);
            __syncthreads();

            /* copy ghost cells along 12 edges  */
            if (i == 0) {
                if (j == 0) 
                    s_val[i][j][k + 1] = model.at(x - 1, y - 1, z);
                if (j == jmax)
                    s_val[i][j + 2][k + 1] = model.at(x - 1, y + 1, z);
            }
            if (i == imax) {
                if (j == 0) 
                    s_val[i + 2][j][k + 1] = model.at(x + 1, y - 1, z);
                if (j == jmax)
                    s_val[i + 2][j + 2][k + 1] = model.at(x + 1, y + 1, z);
            }
            __syncthreads();

            if (j == 0) {
                if (k == 0)
                    s_val[i + 1][j][k] = model.at(x, y - 1, z - 1);
                if (k == kmax)
                    s_val[i + 1][j][k + 2] = model.at(x, y - 1, z + 1);
            }
            if (j == jmax) {
                if (k == 0)
                    s_val[i + 1][j + 2][k] = model.at(x, y + 1, z - 1);
                if (k == kmax)
                    s_val[i + 1][j + 2][k + 2] = model.at(x, y + 1, z + 1);
            }
            __syncthreads();

            // copy ghost-cells along y-direction
            if (k == 0) {
                if (i == 0)
                    s_val[i][j + 1][k] = model.at(x - 1, y, z - 1);
                if (i == imax)
                    s_val[i + 2][j + 1][k] = model.at(x + 1, y, z - 1);
            }
            if (k == kmax) {
                if (i == 0) 
                    s_val[i][j + 1][k + 2] = model.at(x - 1, y, z + 1);
                if (i == imax)
                    s_val[i + 2][j + 1][k + 2] = model.at(x + 1, y, z + 1);
            }
            __syncthreads();

            /*  copy  ghost cells along 16 corners */
            if (k == 0) {
                if (j == 0) {
                    if (i == 0)
                        s_val[i][j][k] = model.at(x - 1, y - 1, z - 1);
                    if (i == imax) {
                        s_val[i + 2][j][k] = model.at(x + 1, y - 1, z - 1);
                    }
                }
                if (j == jmax) {
                    if (i == 0)
                        s_val[i][j + 2][k] = model.at(x - 1, y + 1, z - 1);
                    if (i == imax)
                        s_val[i + 2][j + 2][k] = model.at(x + 1, y + 1, z - 1);
                }
            }
            if (k == kmax) {
                if (j == 0) {
                    if (i == 0)
                        s_val[i][j][k + 2] = model.at(x - 1, y - 1, z + 1);
                    if (i == imax)
                        s_val[i + 2][j][k + 2] = model.at(x + 1, y - 1, z + 1);
                }
                if (j == jmax) {
                    if (i == 0)
                        s_val[i][j + 2][k + 2] = model.at(x - 1, y + 1, z + 1);
                    if (i == imax)
                        s_val[i + 2][j + 2][k + 2] = model.at(x + 1, y + 1, z + 1);
                }
            }
            __syncthreads();

            float v = s_val[i + 1][j + 1][k + 1];
            float temp = 0.f;
            for (int ix = 0; ix < 3; ix++)
                for (int iy = 0; iy < 3; iy++)
                    for (int iz = 0; iz < 3; iz++)
                        temp += weight(ix, iy, iz) * deriv_potFCN(v - s_val[i + ix][j + iy][k + iz], p, sigma);
            objfn(x, y, z) += temp;
        }
    }

    __global__ void hessian_zero_kernel(dev_arrayf hessian, float sigma) {

        int3 idx = Index3D();
        dim3_t dims = hessian.dims();

        int x = idx.x;
        int y = idx.y;
        int z = idx.z;
        if (idx < dims) {
            float temp = 0.f;
            for (int ix = 0; ix < 3; ix++)
                for (int iy = 0; iy < 3; iy++)
                    for (int iz = 0; iz < 3; iz++) temp += weight(ix, iy, iz) * second_deriv_potFunc_zero(sigma);
            hessian(x, y, z) += temp;
        }
    }

    void add_total_var(dev_arrayf model, dev_arrayf objfn, float p, float sigma, cudaStream_t stream) {

        // CUDA kernel parameters
        Grid grid(objfn.dims());

        // update gradients of objective function inplace
        tvd_update_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(model, objfn, p, sigma);
    }

    void calcHessian(dev_arrayf hessian, float sigma, cudaStream_t stream) {

        // CUDA kernel parameters
        Grid grid(hessian.dims());

        // update hessain inplace
        hessian_zero_kernel<<< grid.blocks(), grid.threads(), 0, stream >>>(hessian, sigma);
    }

} // namespace tomocam
