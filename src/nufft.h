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

#include <iostream>

#include <cuda.h>
#include <cufinufft.h>

#include "common.h"
#include "dev_array.h"
#include "gpu/utils.cuh"

// clang-format off
#ifndef NUFFT__H
#define NUFFT__H

#define NUFFT_CALL(ans){ nufft_check((ans), __FILE__, __LINE__); }
// clang-format on

inline void nufft_check(int code, const char *file, int line) {
    if (code != 0) {
        std::cerr << "nufft error at: " << file << ":" << line << std::endl;
        exit(code);
    }
}


namespace tomocam {
    namespace NUFFT {

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
            Grid(int nproj, int npixel, const T *angles, int id) :
                num_projs_(nproj), num_pixels_(npixel), device_id_(id) {

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
                SAFE_CALL(
                    cudaMemcpy(x_, g.x_, bytes, cudaMemcpyDeviceToDevice));
                SAFE_CALL(
                    cudaMemcpy(y_, g.y_, bytes, cudaMemcpyDeviceToDevice));
            }

            // assignment operator
            Grid &operator=(const Grid &g) {
                if (this != &g) {
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
                    SAFE_CALL(
                        cudaMemcpy(x_, g.x_, bytes, cudaMemcpyDeviceToDevice));
                    SAFE_CALL(
                        cudaMemcpy(y_, g.y_, bytes, cudaMemcpyDeviceToDevice));
                }
                return *this;
            }

            // move constructor
            Grid(Grid &&g) {
                num_projs_ = g.num_projs_;
                num_pixels_ = g.num_pixels_;
                device_id_ = g.device_id_;
                x_ = g.x_;
                y_ = g.y_;
                g.x_ = nullptr;
                g.y_ = nullptr;
            }

            // move assignment
            Grid &operator=(Grid &&g) {
                if (this != &g) {
                    num_projs_ = g.num_projs_;
                    num_pixels_ = g.num_pixels_;
                    device_id_ = g.device_id_;
                    x_ = g.x_;
                    y_ = g.y_;
                    g.x_ = nullptr;
                    g.y_ = nullptr;
                }
                return *this;
            }

            // size of non-uniform grid
            int size() const { return num_projs_ * num_pixels_; }

            // number of projections
            int nprojs() const { return num_projs_; }

            // number of pixels
            int npixels() const { return num_pixels_; }

            // get the device id
            int dev_id() const { return device_id_; }

            // get the x coordinates
            T *x() const { return x_; }

            // get the y coordinates
            T *y() const { return y_; }
        };


        // create a plan for the NUFFT (single precision)
        inline void nufftPlan(dim3_t dims, int type, int iflag, int device_id,
            cufinufftf_plan &plan) {

            int ndim = 2;
            int64_t nmodes[] = {dims.z, dims.z};
            int ntransf = dims.x;
            float tol = 1.1e-8;

            // create cufinufft_opts
            cufinufft_opts opts;
            cufinufft_default_opts(&opts);
            opts.gpu_device_id = device_id;
            NUFFT_CALL(cufinufftf_makeplan(
                type, ndim, nmodes, iflag, ntransf, tol, &plan, &opts));
        }

        // create a plan for the NUFFT (double precision)
        inline void nufftPlan(dim3_t dims, int type, int iflag, int device_id,
            cufinufft_plan &plan) {

            int ndim = 2;
            int64_t nmodes[] = {dims.z, dims.z};
            int ntransf = dims.x;
            double tol = 1.e-15;

            // create cufinufft_opts
            cufinufft_opts opts;
            cufinufft_default_opts(&opts);
            opts.gpu_device_id = device_id;
            NUFFT_CALL(cufinufft_makeplan(
                type, ndim, nmodes, iflag, ntransf, tol, &plan, &opts));
        }

        // 2-dimensional NUFFT from non-uniform to uniform grid (single precision)
        inline void nufft2d1(
            DeviceArraycf &c, DeviceArraycf &fk, const Grid<float> &nugrid) {

            // check if device id is same as the one used for the grid
            int dev_id;
            SAFE_CALL(cudaGetDevice(&dev_id));
            if (dev_id != nugrid.dev_id()) {
                std::cerr << "Device id mismatch" << std::endl;
                exit(1);
            }

            cufinufftf_plan plan;
            int type = 1;
            int iflag = 1;
            nufftPlan(fk.dims(), type, iflag, dev_id, plan);

            // set the non-uniform points
            NUFFT_CALL(cufinufftf_setpts(plan, nugrid.size(), nugrid.x(),
                nugrid.y(), nullptr, 0, nullptr, nullptr, nullptr));

            // cast the data to cuFloatComplex
            cuFloatComplex *nu_data =
                reinterpret_cast<cuFloatComplex *>(c.dev_ptr());
            cuFloatComplex *uniform_data =
                reinterpret_cast<cuFloatComplex *>(fk.dev_ptr());

            // execute the plan
            NUFFT_CALL(cufinufftf_execute(plan, nu_data, uniform_data));
            NUFFT_CALL(cufinufftf_destroy(plan));
        }

        // 2-dimensional NUFFT from non-uniform to uniform grid (double precision)
        inline void nufft2d1(
            DeviceArraycd &c, DeviceArraycd &fk, const Grid<double> &nugrid) {

            // check if device id is same as the one used for the grid
            int dev_id;
            SAFE_CALL(cudaGetDevice(&dev_id));
            if (dev_id != nugrid.dev_id()) {
                std::cerr << "Device id mismatch" << std::endl;
                exit(1);
            }

            cufinufft_plan plan;
            int type = 1;
            int iflag = 1;
            nufftPlan(fk.dims(), type, iflag, dev_id, plan);

            // set the non-uniform points
            NUFFT_CALL(cufinufft_setpts(plan, nugrid.size(), nugrid.x(),
                nugrid.y(), nullptr, 0, nullptr, nullptr, nullptr));

            // cast the data to cuDoubleComplex
            cuDoubleComplex *nu_data =
                reinterpret_cast<cuDoubleComplex *>(c.dev_ptr());
            cuDoubleComplex *uniform_data =
                reinterpret_cast<cuDoubleComplex *>(fk.dev_ptr());

            // execute the plan
            NUFFT_CALL(cufinufft_execute(plan, nu_data, uniform_data));
            NUFFT_CALL(cufinufft_destroy(plan));
        }

        // 2-dimensional NUFFT Uniform -> Non-uniform (single precision)
        inline void nufft2d2(
            DeviceArraycf &c, DeviceArraycf &fk, const Grid<float> &nugrid) {

            // check if device id is same as the one used for the grid
            int dev_id;
            SAFE_CALL(cudaGetDevice(&dev_id));
            if (dev_id != nugrid.dev_id()) {
                std::cerr << "Device id mismatch" << std::endl;
                exit(1);
            }

            // make the plan
            cufinufftf_plan plan;
            int type = 2;
            int iflag = -1;
            nufftPlan(fk.dims(), type, iflag, dev_id, plan);

            // set the non-uniform points
            NUFFT_CALL(cufinufftf_setpts(plan, nugrid.size(), nugrid.x(),
                nugrid.y(), nullptr, 0, nullptr, nullptr, nullptr));

            // cast the data to cuFloatComplex
            cuFloatComplex *nu_data =
                reinterpret_cast<cuFloatComplex *>(c.dev_ptr());
            cuFloatComplex *uniform_data =
                reinterpret_cast<cuFloatComplex *>(fk.dev_ptr());

            // execute the plan
            NUFFT_CALL(cufinufftf_execute(plan, nu_data, uniform_data));
            NUFFT_CALL(cufinufftf_destroy(plan));
        }

        // 2-dimensional NUFFT Uniform -> Non-uniform (double precision)
        inline void nufft2d2(
            DeviceArraycd &c, DeviceArraycd &fk, const Grid<double> &nugrid) {

            // check if device id is same as the one used for the grid
            int dev_id;
            SAFE_CALL(cudaGetDevice(&dev_id));
            if (dev_id != nugrid.dev_id()) {
                std::cerr << "Device id mismatch" << std::endl;
                exit(1);
            }

            // make the plan
            cufinufft_plan plan;
            int type = 2;
            int iflag = -1;
            nufftPlan(fk.dims(), type, iflag, dev_id, plan);

            // set the non-uniform points
            NUFFT_CALL(cufinufft_setpts(plan, nugrid.size(), nugrid.x(),
                nugrid.y(), nullptr, 0, nullptr, nullptr, nullptr));

            // cast the data to cuDoubleComplex
            cuDoubleComplex *nu_data =
                reinterpret_cast<cuDoubleComplex *>(c.dev_ptr());
            cuDoubleComplex *uniform_data =
                reinterpret_cast<cuDoubleComplex *>(fk.dev_ptr());

            // execute the plan
            NUFFT_CALL(cufinufft_execute(plan, nu_data, uniform_data));
            NUFFT_CALL(cufinufft_destroy(plan));
        }

    } // namespace NUFFT
} // namespace tomocam

#endif // NUFFT__H
