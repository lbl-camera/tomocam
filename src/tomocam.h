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
    DArray<T> backproject(DArray<T> &, const std::vector<T> &, int);

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
    DArray<T> project(DArray<T> &, const std::vector<T> &, int);

    /**
     * @brief Compute the gradient of the objective function, given current
     * image estimate and sinogram.
     *
     * @param image  the current image estimate.
     * @param transposed_sinogram the transposed sinogram.
     * @param NUFFT::iGrid copy of NUFFT grid on each GPU
     *
     * @return a tuple containing the gradient and the partial function value
     */
    template <typename T>
    std::tuple<DArray<T>, T> gradient(DArray<T> &, DArray<T> &,
        const std::vector<NUFFT::Grid<T>> &);

    /**
     */
    template <typename T>
    std::tuple<DArray<T>, T> gradient2(DArray<T> &, DArray<T> &,
        const std::vector<PointSpreadFunction<T>> &);

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
    DArray<T> mbir(DArray<T> &, T *, T, T, T, int, T, T, T);

    /**
     * @brief Compute the TV Hessian to estimate Lipschitz constant.
     */
    void add_tv_hessian(DArray<float> &, float);

} // namespace tomocam

#endif // TOMOCAM__H
