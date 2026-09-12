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

#include <array>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "dist_array.h"
#include "machine.h"
#include "scheduler.h"
#include "shipper.h"

#include "gpu/finitdiff.cuh"
#include "gpu/shrinkage.cuh"

namespace tomocam::array {

    /**************************************************************************
     **                              grad_u                                 **
     **************************************************************************/

    // per-device worker: halo'd input partition -> 3 halo-free output partitions
    template <typename T>
    void grad_u_worker(Partition<T> u, Partition<T> dudx, Partition<T> dudy,
                       Partition<T> dudz, int device) {
        DeviceGuard guard(device);

        auto nparts = Machine::config.num_of_partitions(dudx.dims(), dudx.bytes());
        auto sub_u    = create_partitions(u, nparts, 1);   // halo=1, re-subdivide
        auto sub_dudx = create_partitions(dudx, nparts);   // no halo (pure output)
        auto sub_dudy = create_partitions(dudy, nparts);
        auto sub_dudz = create_partitions(dudz, nparts);

        GPUToHost<Partition<T>, DeviceArray<T>> shipper;
        Scheduler<Partition<T>, DeviceArray<T>> scheduler(sub_u);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto &&[idx, d_u] = std::move(work.value());

                // pure-output arrays: allocate directly, no H2D copy-in needed
                DeviceArray<T> d_dudx(sub_dudx[idx].dims());
                DeviceArray<T> d_dudy(sub_dudy[idx].dims());
                DeviceArray<T> d_dudz(sub_dudz[idx].dims());

                gpu::grad_u<T>(d_u, d_dudx, d_dudy, d_dudz);

                shipper.push(sub_dudx[idx], std::move(d_dudx));
                shipper.push(sub_dudy[idx], std::move(d_dudy));
                shipper.push(sub_dudz[idx], std::move(d_dudz));
            }
        }
    }

    // multi-GPU dispatch
    // TODO(MULTIPROC): grad_u is a 1-voxel halo stencil across the x-axis,
    // same as add_total_var2 in tv_update.cpp -- cross-rank halo exchange
    // (sol.update_neigh_proc()) is not threaded through here yet.
    template <typename T>
    std::array<DArray<T>, 3> grad_u(const DArray<T> &u) {
        dim3_t dims = u.dims();
        DArray<T> dudx(dims), dudy(dims), dudz(dims);

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > u.nslices()) nDevice = u.nslices();

        auto p_u    = create_partitions(const_cast<DArray<T> &>(u), nDevice, 1);
        auto p_dudx = create_partitions(dudx, nDevice);
        auto p_dudy = create_partitions(dudy, nDevice);
        auto p_dudz = create_partitions(dudz, nDevice);

        std::vector<std::thread> threads(nDevice);
        for (int i = 0; i < nDevice; i++) {
            threads[i] = std::thread(grad_u_worker<T>, p_u[i], p_dudx[i],
                                     p_dudy[i], p_dudz[i], i);
        }
        Machine::config.barrier();
        for (auto &t : threads) t.join();

        return {std::move(dudx), std::move(dudy), std::move(dudz)};
    }

    /**************************************************************************
     **                             divergence                              **
     **************************************************************************/

    template <typename T>
    void divergence_worker(Partition<T> p, Partition<T> q, Partition<T> r,
                           Partition<T> div, int device) {
        DeviceGuard guard(device);

        auto nparts = Machine::config.num_of_partitions(div.dims(), div.bytes());
        auto sub_p   = create_partitions(p, nparts, 1);
        auto sub_q   = create_partitions(q, nparts, 1);
        auto sub_r   = create_partitions(r, nparts, 1);
        auto sub_div = create_partitions(div, nparts);

        GPUToHost<Partition<T>, DeviceArray<T>> shipper;
        // Scheduler natively prefetches only 1 or 2 host-side input vectors
        // (see src/scheduler.h) -- p & q are prefetched asynchronously, and
        // r's DeviceArray is constructed synchronously per work-item below.
        // Deliberate simplification for this first pass.
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> scheduler(sub_p, sub_q);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto &&[idx, d_p, d_q] = std::move(work.value());
                DeviceArray<T> d_r(sub_r[idx]);               // synchronous H2D
                DeviceArray<T> d_div(sub_div[idx].dims());     // pure output

                gpu::divergence<T>(d_p, d_q, d_r, d_div);

                shipper.push(sub_div[idx], std::move(d_div));
            }
        }
    }

    // TODO(MULTIPROC): same cross-rank halo-exchange gap as grad_u above.
    template <typename T>
    DArray<T> divergence(const std::array<DArray<T>, 3> &d) {
        dim3_t dims = d[0].dims();
        DArray<T> div(dims);

        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > div.nslices()) nDevice = div.nslices();

        auto p_p   = create_partitions(const_cast<DArray<T> &>(d[0]), nDevice, 1);
        auto p_q   = create_partitions(const_cast<DArray<T> &>(d[1]), nDevice, 1);
        auto p_r   = create_partitions(const_cast<DArray<T> &>(d[2]), nDevice, 1);
        auto p_div = create_partitions(div, nDevice);

        std::vector<std::thread> threads(nDevice);
        for (int i = 0; i < nDevice; i++) {
            threads[i] = std::thread(divergence_worker<T>, p_p[i], p_q[i],
                                     p_r[i], p_div[i], i);
        }
        Machine::config.barrier();
        for (auto &t : threads) t.join();
        return div;
    }

    /**************************************************************************
     **                             laplacian                               **
     **************************************************************************/

    // laplacian(u) = div(grad(u)), implemented as a composition of the two
    // passes above rather than a fused kernel -- a reasonable optimization
    // to consider later, once this version is verified.
    template <typename T>
    DArray<T> laplacian(const DArray<T> &u) {
        return divergence<T>(grad_u<T>(u));
    }

    /**************************************************************************
     **                             shrinkage                               **
     **************************************************************************/

    // per-device worker: 6 arrays in (dx,dy,dz,bx,by,bz), 6 out (bx,by,bz
    // mutated in place, dcx,dcy,dcz pure output). Purely elementwise (no
    // stencil), so unlike grad_u_worker/divergence_worker no halo is needed
    // anywhere.
    //
    // Scheduler natively prefetches only 1 or 2 host-side input vectors
    // (see src/scheduler.h) -- dx & bx are prefetched asynchronously, and
    // dy,dz,by,bz are constructed synchronously per work-item below, same
    // simplification divergence_worker uses for its 3rd input.
    template <typename T>
    void shrinkage_worker(Partition<T> dx, Partition<T> dy, Partition<T> dz,
                          Partition<T> bx, Partition<T> by, Partition<T> bz,
                          Partition<T> dcx, Partition<T> dcy, Partition<T> dcz,
                          T lambda_mu, T epsilon, int device) {
        DeviceGuard guard(device);

        auto nparts = Machine::config.num_of_partitions(dx.dims(), dx.bytes());
        auto sub_dx  = create_partitions(dx, nparts);
        auto sub_dy  = create_partitions(dy, nparts);
        auto sub_dz  = create_partitions(dz, nparts);
        auto sub_bx  = create_partitions(bx, nparts);
        auto sub_by  = create_partitions(by, nparts);
        auto sub_bz  = create_partitions(bz, nparts);
        auto sub_dcx = create_partitions(dcx, nparts);
        auto sub_dcy = create_partitions(dcy, nparts);
        auto sub_dcz = create_partitions(dcz, nparts);

        GPUToHost<Partition<T>, DeviceArray<T>> shipper;
        Scheduler<Partition<T>, DeviceArray<T>, DeviceArray<T>> scheduler(sub_dx, sub_bx);
        while (scheduler.has_work()) {
            auto work = scheduler.get_work();
            if (work.has_value()) {
                auto &&[idx, d_dx, d_bx] = std::move(work.value());
                DeviceArray<T> d_dy(sub_dy[idx]);   // synchronous H2D
                DeviceArray<T> d_dz(sub_dz[idx]);
                DeviceArray<T> d_by(sub_by[idx]);
                DeviceArray<T> d_bz(sub_bz[idx]);
                DeviceArray<T> d_dcx(sub_dcx[idx].dims()); // pure output
                DeviceArray<T> d_dcy(sub_dcy[idx].dims());
                DeviceArray<T> d_dcz(sub_dcz[idx].dims());

                gpu::shrinkage<T>(d_dx, d_dy, d_dz, d_bx, d_by, d_bz, d_dcx,
                    d_dcy, d_dcz, lambda_mu, epsilon);

                // bregman_b is mutated in place: ship the same (now-updated)
                // device buffer back onto the same host partition it came
                // from -- the total_var2 in-place pattern (tv_update.cpp),
                // scaled to 3 components.
                shipper.push(sub_bx[idx], std::move(d_bx));
                shipper.push(sub_by[idx], std::move(d_by));
                shipper.push(sub_bz[idx], std::move(d_bz));
                shipper.push(sub_dcx[idx], std::move(d_dcx));
                shipper.push(sub_dcy[idx], std::move(d_dcy));
                shipper.push(sub_dcz[idx], std::move(d_dcz));
            }
        }
    }

    // multi-GPU dispatch
    template <typename T>
    void shrinkage(const std::array<DArray<T>, 3> &dx,
                  std::array<DArray<T>, 3> &bregman_b,
                  std::array<DArray<T>, 3> &d, T lambda_mu, T epsilon) {

        dim3_t dims = dx[0].dims();
        int nDevice = Machine::config.num_of_gpus();
        if (nDevice > dims.x) nDevice = dims.x;

        auto p_dx  = create_partitions(const_cast<DArray<T> &>(dx[0]), nDevice);
        auto p_dy  = create_partitions(const_cast<DArray<T> &>(dx[1]), nDevice);
        auto p_dz  = create_partitions(const_cast<DArray<T> &>(dx[2]), nDevice);
        auto p_bx  = create_partitions(bregman_b[0], nDevice);
        auto p_by  = create_partitions(bregman_b[1], nDevice);
        auto p_bz  = create_partitions(bregman_b[2], nDevice);
        auto p_dcx = create_partitions(d[0], nDevice);
        auto p_dcy = create_partitions(d[1], nDevice);
        auto p_dcz = create_partitions(d[2], nDevice);

        std::vector<std::thread> threads(nDevice);
        for (int i = 0; i < nDevice; i++) {
            threads[i] = std::thread(shrinkage_worker<T>, p_dx[i], p_dy[i],
                p_dz[i], p_bx[i], p_by[i], p_bz[i], p_dcx[i], p_dcy[i],
                p_dcz[i], lambda_mu, epsilon, i);
        }
        Machine::config.barrier();
        for (auto &t : threads) t.join();
    }

    // explicit instantiation
    template std::array<DArray<float>, 3> grad_u(const DArray<float> &);
    template std::array<DArray<double>, 3> grad_u(const DArray<double> &);
    template DArray<float> divergence(const std::array<DArray<float>, 3> &);
    template DArray<double> divergence(const std::array<DArray<double>, 3> &);
    template DArray<float> laplacian(const DArray<float> &);
    template DArray<double> laplacian(const DArray<double> &);
    template void shrinkage(const std::array<DArray<float>, 3> &,
        std::array<DArray<float>, 3> &, std::array<DArray<float>, 3> &, float,
        float);
    template void shrinkage(const std::array<DArray<double>, 3> &,
        std::array<DArray<double>, 3> &, std::array<DArray<double>, 3> &,
        double, double);

} // namespace tomocam::array
