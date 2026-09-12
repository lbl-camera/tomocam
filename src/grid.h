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
#ifndef NUFFT_GRID_H
#define NUFFT_GRID_H

#include "common.h"
#include "dev_array.h"
#include "gpu/utils.cuh"
#include "machine.h"
#include <cuda.h>

namespace tomocam::nufft {

    /*! \brief A class to store the non-uniform grid points on the device
     *
     *  The class is templated on the type of the grid points (float or
     * double). The class is used to store the polar grid points on the
     * device. One instance of the class is created for each device.
     */
    template <typename T>
    class Grid {
      private:
        int device_id_;
        int num_projs_;
        int num_pixels_;
        gpuMem::cuniquePtr<T> x_;
        gpuMem::cuniquePtr<T> y_;

      public:
        // default constructor
        explicit Grid()
            : x_(nullptr), y_(nullptr), device_id_(-1), num_projs_(0),
              num_pixels_(0) {}

        /*! \brief Constructor to create the non-uniform grid on the device
         *
         *  The constructor creates the non-uniform (polar) grid on the
         * device. The grid is created using the make_nugrid function in the
         * gpu namespace. The grid is created for a given number of
         * projections, number of pixels per detector row, and the angles of
         * the projections.
         *
         *  \param nproj Number of projections
         *  \param npixel Number of pixels per detector row
         *  \param angles Angles of the projections
         *  \param id Device id
         */
        explicit Grid(int nproj, int npixel, const T *angles, int id)
            : num_projs_(nproj), num_pixels_(npixel), device_id_(id) {

            // set device
            DeviceGuard guard(device_id_);

            // allocate memory for the non-uniform points on the device
            size_t npts = num_projs_ * num_pixels_;
            x_ = gpuMem::make_cuniquePtr<T>(npts);
            y_ = gpuMem::make_cuniquePtr<T>(npts);
            gpu::make_nugrid<T>(num_pixels_, num_projs_, x_.get(), y_.get(), angles);
            SAFE_CALL(cudaGetLastError()); // Check for kernel launch errors
            SAFE_CALL(cudaDeviceSynchronize());
        }

        // delete copy constructor and assignment operator
        Grid(const Grid &g) = delete;
        Grid &operator=(const Grid &g) = delete;

        // move constructor
        Grid(Grid &&g) noexcept
            : num_projs_(g.num_projs_), num_pixels_(g.num_pixels_),
              device_id_(g.device_id_) {
            x_ = std::move(g.x_);
            y_ = std::move(g.y_);
        }

        // move assignment operator
        Grid &operator=(Grid &&g) noexcept {
            if (this != &g) {
                num_projs_ = g.num_projs_;
                num_pixels_ = g.num_pixels_;
                device_id_ = g.device_id_;
                x_ = std::move(g.x_);
                y_ = std::move(g.y_);
            }
            return *this;
        }

        // getters
        int nprojs() const { return num_projs_; }
        int size() const { return num_projs_ * num_pixels_; }
        int npixels() const { return num_pixels_; }
        int dev_id() const { return device_id_; }
        T *x() const { return x_.get(); }
        T *y() const { return y_.get(); }
    };
} // namespace tomocam::nufft

#endif // NUFFT_GRID_H
