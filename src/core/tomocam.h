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

#ifndef TOMOCAM__H
#define TOMOCAM__H

#include <optional>
#include <tuple>
#include <vector>

#include "memory/dist_array.h"
#include "transforms/nufft.h"
#include "transforms/toeplitz.h"
#include "utils/timer.h"

template <typename T>
using psf_t = tomocam::transforms::PointSpreadFunction<T>;

namespace tomocam {

    /**
     * @brief Compute the backprojection of a sinogram.
     *
     * @param sinogram The sinogram to backproject.
     * @param angles The angles of the sinogram.
     * @param flag boolean flag to indicate whether to apply the ramp filter.
     *
     * @return The backprojected image.
     */
    template <typename T>
    DArray<T> backproject(DArray<T> &sinogram, const std::vector<T> &angles,
                          bool flag = false);

    /**
     * @brief Compute the forward projection of an image.
     *
     * @param image The image to project.
     * @param angles The angles of the sinogram.
     *
     * @return The forward projected sinogram.
     */
    template <typename T>
    DArray<T> project(DArray<T> &image, const std::vector<T> &angles);

    /**
     * @brief Compute the gradient of the objective function, given current
     * image estimate and sinogram.
     *
     * @param current_solution current solution.
     * @param transposed_sinogram the transposed sinogram.
     * @param grids A copy of NUFFT Grid on each GPU
     *
     * @return a tuple containing the gradient and the partial function value
     */
    template <typename T>
    DArray<T> gradient(DArray<T> &current_solution, DArray<T> &transposed_sinogram,
                       const std::vector<transforms::NUFFT::Grid<T>> &grids);

    /**
     * @brief Compute the gradient of the objective function, given current
     * image estimate and sinogram.
     *
     * @param current_solution current solution.
     * @param transposed_sinogram the transposed sinogram.
     * @param psf A copy of PSF on each GPU device.
     *
     * @return a tuple containing the gradient and the partial function value
     */

    template <typename T>
    DArray<T> gradient2(DArray<T> &current_solution, DArray<T> &transposed_sinogram,
                        const std::vector<psf_t<T>> &psf);

    /**
     * @brief Compute the value of the objective function, given current
     * solution
     *
     * @param current_solution current solution.
     * @param sinogram sinogram.
     * @param grids std::vector of NUFFT::Grid types per device
     *
     * @return the value of the objective function
     */
    template <typename T>
    T residual(DArray<T> &current_solution, DArray<T> &sinogram,
               const std::vector<transforms::NUFFT::Grid<T>> &grids);

    /**
     * @brief Compute the value of the objective function, given current
     * solution
     *
     * @param current_solution current solution.
     * @param transposed_sinogram the transposed sinogram.
     * @param psf A copy of PSF on each GPU device.
     * @param sino_norm dot product of sinogram with itself.
     *
     * @return the value of the objective function
     */
    template <typename T>
    T residual2(DArray<T> &current_solution, DArray<T> &transposed_sinogram,
                const std::vector<psf_t<T>> &psf, T sino_norm);

    /**
     * @brief Compute TV penalty and update gradients in-place.
     *
     * @param image The current image estimate to compute the TV penalty.
     * @param grad The gradient to update.
     * @param beta The TV penalty parameter.
     * @param eps The TV penalty epsilon.
     */
    namespace reconstruction {
        template <typename T>
        void add_total_var2(DArray<T> &image, DArray<T> &grad, T beta, T eps);

        /**
         * @brief Compute the MBIR reconstruction using Toeplitz matrix.
         *
         * @param initial_guess initial guess
         * @param sinogram The sinogram to reconstruct.
         * @param angles The angles of the sinogram.
         * @param center The center of rotation.
         * @param num_iter The number of iterations.
         * @param sigma The regularization parameter.
         * @param tolerance The stopping criterion.
         * @param xtol The tolerance for the solution.
         */
        template <typename T>
        DArray<T> mbir2(const DArray<T> &initial_guess, const DArray<T> &sinogram,
                        std::vector<T> angles, T center, int num_iter, T sigma,
                        T tolerance, T xtol);

        /**
         * @brief Compute the MBIR reconstruction.
         *
         * @param initial_guess initial guess
         * @param sinogram The sinogram to reconstruct.
         * @param angles The angles of the sinogram.
         * @param center The center of rotation.
         * @param num_iter The number of iterations.
         * @param sigma The regularization parameter.
         * @param tolerance The stopping criterion.
         * @param xtol The tolerance for the solution.
         */
        template <typename T>
        DArray<T> mbir(DArray<T> &initial_guess, DArray<T> &sinogram,
                       std::vector<T> angles, T center, int num_iter, T sigma,
                       T tolerance, T xtol);
    } // namespace reconstruction

    namespace preprocessing {
        /**
         * @brief Zero pad the sinogram by a factor of \f$\sqrt{2}\f$
         * @param sinogram sinogram
         * @param factor padding factor
         *
         * @return zero padded sinogram
         */
        template <typename T>
        DArray<T> preproc(DArray<T> &sinogram, T factor);

        /**
         * @brief Crop the reconstruction by a factor of \f$\sqrt{2}\f$
         * @param reconstruction reconstruction
         * @param crop_size size to crop by
         *
         * @return cropped reconstruction
         */
        template <typename T>
        DArray<T> postproc(DArray<T> &reconstruction, int crop_size);

        /**
         * @brief Pad a 2D array to the next power of two in each dimension.
         * @param array The 2D array to pad.
         * @param pad_size The size by which to pad the array.
         * @param pad_type Direction of padding (PadType::LEFT, PadType::RIGHT,
         * PadType::SYMMETRIC).
         */
        template <typename T>
        DArray<T> pad2d(DArray<T> &array, int pad_size, PadType pad_type);
    } // namespace preprocessing

    /**
     * @brief compute the norm of the difference between two reconstructions
     * @param x1 first reconstruction
     * @param x2 second reconstruction
     *
     * @return the norm of the difference
     */
    template <typename T>
    T xerror(DArray<T> &x1, DArray<T> &x2);

} // namespace tomocam

#endif // TOMOCAM__H
