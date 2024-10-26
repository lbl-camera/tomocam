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

#include <tuple>
#include <vector>

#include "dist_array.h"
#include "nufft.h"
#include "toeplitz.h"

namespace tomocam {

    /**
     * @brief Compute the backprojection of a sinogram.
     *
     * @param sinogram The sinogram to backproject.
     * @param angles The angles of the sinogram.
     * @param center The center of rotation.
     *
     * @return The backprojected image.
     */
    template <typename T>
    DArray<T> backproject(DArray<T> &, const std::vector<T> &, T);

    /**
     * @brief Compute the forward projection of an image.
     *
     * @param image The image to project.
     * @param angles The angles of the sinogram.
     * @param center The center of rotation.
     *
     * @return The forward projected sinogram.
     */
    template <typename T>
    DArray<T> project(DArray<T> &, const std::vector<T> &, T);

    /**
     * @brief Compute the gradient of the objective function, given current
     * image estimate and sinogram.
     *
     * @param current solution.
     * @param transposed_sinogram the transposed sinogram.
     * @param A copy of NUFFT Grid on each GPU
     *
     * @return a tuple containing the gradient and the partial function value
     */
    template <typename T>
    DArray<T> gradient(DArray<T> &, DArray<T> &,
        const std::vector<NUFFT::Grid<T>> &, T);

    /**
     * @brief Compute the gradient of the objective function, given current
     * image estimate and sinogram.
     *
     * @param current solution.
     * @param transposed_sinogram the transposed sinogram.
     * @param A copy of PSF on each GPU device.
     *
     * @return a tuple containing the gradient and the partial function value
     */

    template <typename T>
    DArray<T> gradient2(DArray<T> &, DArray<T> &,
        const std::vector<PointSpreadFunction<T>> &);

    /**
     * @brief Compute the value of the objective function, given current
     * solution
     *
     * @param current solution.
     * @param std::vector of NUFFT::Grid types per device
     * @param center of rotation
     *
     * @return the value of the objective function
     */
    template <typename T>
    T function_value(DArray<T> &, DArray<T> &,
        const std::vector<NUFFT::Grid<T>> &, T);

    /**
     * @brief Compute the value of the objective function, given current
     * solution
     *
     * @param current solution.
     * @param transposed_sinogram the transposed sinogram.
     * @param dot product of sinograom with itself.
     *
     * @return the value of the objective function
     */
    template <typename T>
    T function_value2(DArray<T> &, DArray<T> &,
        const std::vector<PointSpreadFunction<T>> &, T);

    /**
     * @brief Compute TV penalty and update gradients in-place.
     *
     * @param image The current image estimate to compute the TV penalty.
     * @param grad The gradient to update.
     * @param beta The TV penalty parameter.
     * @param eps The TV penalty epsilon.
     */
    template <typename T>
    void add_total_var(DArray<T> &, DArray<T> &, float, float);

    template <typename T>
    DArray<T> mbir2(DArray<T> &, std::vector<T>, T, T, T, int, T, T, T);

    template <typename T>
    DArray<T> mbir(DArray<T> &, std::vector<T>, T, T, T, int, T, T, T);

    /**
     * @brief Compute the TV Hessian to estimate Lipschitz constant.
     */
    namespace gpu {
        template <typename T>
        void add_tv_hessian(DArray<T> &, float);
    }

    /**
     * @brief classical gradeint calculation
     *
     *  @param current solution.
     *  @param sinogram
     *  @param NUFFT object for each GPU device
     *  @param center of rotation
     *
     *  @return the gradient
     */
    template <typename T>
    DArray<T> gradient(DArray<T> &, DArray<T> &,
        const std::vector<NUFFT::Grid<T>> &, int);

    /**
     * @brief Compute the value of the objective function, given current
     * solution
     *
     * @param current solution.
     * @param sinogram
     * @param vector of NUFFT objects for each GPU device
     * @param center of rotation
     *
     * @return the value of the objective function
     */
    template <typename T>
    T function_value(DArray<T> &, DArray<T> &,
        const std::vector<NUFFT::Grid<T>> &, int);

    /**
     * @brief Zero pad the sinogram by a factor of \sqrt{2}
     * @param sinogram
     *
     * @return zero padded sinogram
     */
    template <typename T>
    DArray<T> preproc(DArray<T> &, T);

    /**
     * @brief Crop the reconstruction by a factor of \sqrt{2}
     * @param reconstruction
     *
     * @return cropped reconstruction
     */
    template <typename T>
    DArray<T> postproc(DArray<T> &, int);

} // namespace tomocam

#endif // TOMOCAM__H
