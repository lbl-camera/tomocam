
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
 * the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "gpu/utils.cuh"

#include "gpu/finitdiff.cuh"

// NOTE: deliberately does NOT include "potential_function.cuh". That header
// declares its own NX/NY/NZ constants in this same tomocam::gpu namespace,
// which would collide with the NX/NY/NZ below (ODR/redefinition), and none
// of its actual contents (FILTER/weight/potfunc/d_pot_func) are used here.

namespace tomocam::gpu {

    constexpr int NX = 1;
    constexpr int NY = 16;
    constexpr int NZ = 32;

    /********************************************************************************
     **               grad_u: forward-difference gradient of a 3D array           **
     ********************************************************************************/
    template <typename T>
    __global__ void grad_u_kernel(DeviceMemory<T> u, DeviceMemory<T> dudx,
                                  DeviceMemory<T> dudy, DeviceMemory<T> dudz) {

        auto idx = Index3D();
        if (idx < dudx.dims()) {
            T v = u.at(idx.x, idx.y, idx.z);
            dudx[idx] = u.at(idx.x + 1, idx.y,     idx.z    ) - v;
            dudy[idx] = u.at(idx.x,     idx.y + 1, idx.z    ) - v;
            dudz[idx] = u.at(idx.x,     idx.y,     idx.z + 1) - v;
        }
    }

    template <typename T>
    void grad_u(const DeviceArray<T> &u, DeviceArray<T> &dudx,
               DeviceArray<T> &dudy, DeviceArray<T> &dudz) {

        auto dims = dudx.dims();

        // CUDA kernel parameters
        dim3 block(NZ, NY, NX);
        dim3 grid;
        grid.x = (dims.z + NZ - 1) / NZ;
        grid.y = (dims.y + NY - 1) / NY;
        grid.z = (dims.x + NX - 1) / NX;

        grad_u_kernel<T><<<grid, block>>>(u, dudx, dudy, dudz);
        SAFE_CALL(cudaGetLastError());
    }

    /******************************************************************************
     **     divergence: backward-difference divergence (adjoint of grad_u)      **
     ******************************************************************************/
    // grad_u_kernel's last-slice-per-axis forward difference is forced to 0
    // (a Neumann boundary: reading past the end clamps back to the same
    // value, so u.at(N,...)-u.at(N-1,...)==0), which means the *true* matrix
    // transpose of grad_u also zeroes the input component at the last index
    // per axis -- not just the "previous" term at the first index. Both
    // halves must be dropped (not clamped) to be the exact adjoint (verified
    // by test/finitdiff.cpp's adjoint check, which needs no boundary
    // correction once both halves are handled this way).
    //
    // The x-axis is the one axis this codebase ever partitions across GPUs
    // (with real halo data at interior partition seams), so a local
    // idx.x==0/dims.x-1 boundary is only the TRUE global domain edge when
    // this partition's own halo on that side is 0 -- p.halo_lo()/halo_hi()
    // distinguish "true edge" from "interior seam with a real neighbor".
    // y/z are never partitioned, so their local edges are always the true
    // global edges and need no such check.
    template <typename T>
    __global__ void divergence_kernel(DeviceMemory<T> p, DeviceMemory<T> q,
                                      DeviceMemory<T> r, DeviceMemory<T> div) {

        auto idx = Index3D();
        if (idx < div.dims()) {
            auto dims = div.dims();
            bool x_true_first = (idx.x == 0)         && (p.halo_lo() == 0);
            bool x_true_last  = (idx.x == dims.x - 1) && (p.halo_hi() == 0);
            T p_here = x_true_last  ? T(0) : p.at(idx.x, idx.y, idx.z);
            T p_prev = x_true_first ? T(0) : p.at(idx.x - 1, idx.y, idx.z);
            T q_here = (idx.y < dims.y - 1) ? q.at(idx.x, idx.y, idx.z) : T(0);
            T q_prev = (idx.y > 0)          ? q.at(idx.x, idx.y - 1, idx.z) : T(0);
            T r_here = (idx.z < dims.z - 1) ? r.at(idx.x, idx.y, idx.z) : T(0);
            T r_prev = (idx.z > 0)          ? r.at(idx.x, idx.y, idx.z - 1) : T(0);
            div[idx] = (p_here - p_prev) + (q_here - q_prev) + (r_here - r_prev);
        }
    }

    template <typename T>
    void divergence(const DeviceArray<T> &p, const DeviceArray<T> &q,
                    const DeviceArray<T> &r, DeviceArray<T> &div) {

        auto dims = div.dims();

        // CUDA kernel parameters
        dim3 block(NZ, NY, NX);
        dim3 grid;
        grid.x = (dims.z + NZ - 1) / NZ;
        grid.y = (dims.y + NY - 1) / NY;
        grid.z = (dims.x + NX - 1) / NX;

        divergence_kernel<T><<<grid, block>>>(p, q, r, div);
        SAFE_CALL(cudaGetLastError());
    }

    // explicit instantiation
    template void grad_u(const DeviceArray<float> &, DeviceArray<float> &,
        DeviceArray<float> &, DeviceArray<float> &);
    template void grad_u(const DeviceArray<double> &, DeviceArray<double> &,
        DeviceArray<double> &, DeviceArray<double> &);
    template void divergence(const DeviceArray<float> &, const DeviceArray<float> &,
        const DeviceArray<float> &, DeviceArray<float> &);
    template void divergence(const DeviceArray<double> &, const DeviceArray<double> &,
        const DeviceArray<double> &, DeviceArray<double> &);

} // namespace tomocam::gpu
