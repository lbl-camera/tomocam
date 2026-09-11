/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "gpu/utils.cuh"

#include "gpu/shrinkage.cuh"

// NOTE: deliberately does NOT include "potential_function.cuh" -- that
// header declares its own NX/NY/NZ constants in this same tomocam::gpu
// namespace, which would collide with the NX/NY/NZ below (ODR/redefinition),
// same reasoning as gpu/finitdiff.cu.

namespace tomocam::gpu {

    constexpr int NX = 1;
    constexpr int NY = 16;
    constexpr int NZ = 32;

    /**************************************************************************
     ** shrinkage: isotropic TV vector shrinkage + Bregman update, per-voxel **
     **************************************************************************/
    template <typename T>
    __global__ void shrinkage_kernel(DeviceMemory<T> dx, DeviceMemory<T> dy,
        DeviceMemory<T> dz, DeviceMemory<T> bx, DeviceMemory<T> by,
        DeviceMemory<T> bz, DeviceMemory<T> dcx, DeviceMemory<T> dcy,
        DeviceMemory<T> dcz, T lambda_mu, T epsilon) {

        auto idx = Index3D();
        if (idx < dx.dims()) {
            T vx = dx[idx] + bx[idx];
            T vy = dy[idx] + by[idx];
            T vz = dz[idx] + bz[idx];
            T sk = sqrt(vx * vx + vy * vy + vz * vz);
            T shrink = max(T(0), sk - lambda_mu) / (sk + epsilon);

            T dcx_v = shrink * vx;
            T dcy_v = shrink * vy;
            T dcz_v = shrink * vz;

            bx[idx] += dx[idx] - dcx_v;
            by[idx] += dy[idx] - dcy_v;
            bz[idx] += dz[idx] - dcz_v;

            dcx[idx] = dcx_v;
            dcy[idx] = dcy_v;
            dcz[idx] = dcz_v;
        }
    }

    template <typename T>
    void shrinkage(const DeviceArray<T> &dx, const DeviceArray<T> &dy,
        const DeviceArray<T> &dz, DeviceArray<T> &bx, DeviceArray<T> &by,
        DeviceArray<T> &bz, DeviceArray<T> &dcx, DeviceArray<T> &dcy,
        DeviceArray<T> &dcz, T lambda_mu, T epsilon) {

        auto dims = dx.dims();

        // CUDA kernel parameters
        dim3 block(NZ, NY, NX);
        dim3 grid;
        grid.x = (dims.z + NZ - 1) / NZ;
        grid.y = (dims.y + NY - 1) / NY;
        grid.z = (dims.x + NX - 1) / NX;

        shrinkage_kernel<T><<<grid, block>>>(dx, dy, dz, bx, by, bz, dcx, dcy,
            dcz, lambda_mu, epsilon);
        SAFE_CALL(cudaGetLastError());
    }

    // explicit instantiation
    template void shrinkage(const DeviceArray<float> &, const DeviceArray<float> &,
        const DeviceArray<float> &, DeviceArray<float> &, DeviceArray<float> &,
        DeviceArray<float> &, DeviceArray<float> &, DeviceArray<float> &,
        DeviceArray<float> &, float, float);
    template void shrinkage(const DeviceArray<double> &, const DeviceArray<double> &,
        const DeviceArray<double> &, DeviceArray<double> &, DeviceArray<double> &,
        DeviceArray<double> &, DeviceArray<double> &, DeviceArray<double> &,
        DeviceArray<double> &, double, double);

} // namespace tomocam::gpu
