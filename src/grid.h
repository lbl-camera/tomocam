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
#include "gpu/utils.cuh"
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
        T *x_;
        T *y_;

      public:
        // default constructor
        Grid() : x_(nullptr), y_(nullptr) {}

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
        Grid(int nproj, int npixel, const T *angles, int id)
            : num_projs_(nproj), num_pixels_(npixel), device_id_(id) {

            // set device
            SAFE_CALL(cudaSetDevice(device_id_));

            // allocate memory for the non-uniform points on the device
            size_t bytes = sizeof(T) * num_projs_ * num_pixels_;
            SAFE_CALL(cudaMalloc(&x_, bytes));
            SAFE_CALL(cudaMalloc(&y_, bytes));
            gpu::make_nugrid<T>(num_pixels_, num_projs_, x_, y_, angles);
            cudaDeviceSynchronize();
        }

        // destructor
        ~Grid() {
            SAFE_CALL(cudaSetDevice(device_id_));
            if (x_) SAFE_CALL(cudaFree(x_));
            if (y_) SAFE_CALL(cudaFree(y_));
        }

        // copy constructor
        Grid(const Grid &g) {
            num_projs_ = g.num_projs_;
            num_pixels_ = g.num_pixels_;
            device_id_ = g.device_id_;

            // set device
            SAFE_CALL(cudaSetDevice(device_id_));

            // allocate memory for the non-uniform points on the device
            size_t bytes = num_projs_ * num_pixels_ * sizeof(T);
            SAFE_CALL(cudaMalloc(&x_, bytes));
            SAFE_CALL(cudaMalloc(&y_, bytes));

            // copy the data
            SAFE_CALL(cudaMemcpy(x_, g.x_, bytes, cudaMemcpyDeviceToDevice));
            SAFE_CALL(cudaMemcpy(y_, g.y_, bytes, cudaMemcpyDeviceToDevice));
        }

        // assignment operator
        Grid &operator=(const Grid &g) {
            if (this != &g) {
                num_projs_ = g.num_projs_;
                num_pixels_ = g.num_pixels_;
                device_id_ = g.device_id_;

                // set device
                SAFE_CALL(cudaSetDevice(device_id_));

                // free the memory if it is already allocated
                if (x_) SAFE_CALL(cudaFree(x_));
                if (y_) SAFE_CALL(cudaFree(y_));

                // allocate memory for the non-uniform points on the device
                size_t bytes = num_projs_ * num_pixels_ * sizeof(T);
                SAFE_CALL(cudaMalloc(&x_, bytes));
                SAFE_CALL(cudaMalloc(&y_, bytes));

                // copy the data
                SAFE_CALL(cudaMemcpy(x_, g.x_, bytes, cudaMemcpyDeviceToDevice));
                SAFE_CALL(cudaMemcpy(y_, g.y_, bytes, cudaMemcpyDeviceToDevice));
            }
            return *this;
        }

        // move constructor
        Grid(Grid &&g) noexcept {
            num_projs_ = g.num_projs_;
            num_pixels_ = g.num_pixels_;
            device_id_ = g.device_id_;
            x_ = g.x_;
            y_ = g.y_;
            g.x_ = nullptr;
            g.y_ = nullptr;
        }

        // move assignment operator
        Grid &operator=(Grid &&g) noexcept {
            if (this != &g) {
                num_projs_ = g.num_projs_;
                num_pixels_ = g.num_pixels_;
                device_id_ = g.device_id_;

                // free the memory if it is already allocated
                if (x_) SAFE_CALL(cudaFree(x_));
                if (y_) SAFE_CALL(cudaFree(y_));

                x_ = g.x_;
                y_ = g.y_;
                g.x_ = nullptr;
                g.y_ = nullptr;
            }
            return *this;
        }

        // getters
        int nprojs() const { return num_projs_; }
        int npixels() const { return num_pixels_; }
        int dev_id() const { return device_id_; }
        T *x() const { return x_; }
        T *y() const { return y_; }
        int size() const { return num_projs_ * num_pixels_; }
    };

} // namespace tomocam::nufft

#endif // GRID__H
