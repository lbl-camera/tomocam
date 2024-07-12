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



#ifndef TOMOCAM_INTERNALS__H
#define TOMOCAM_INTERNALS__H

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "dist_array.h"
#include "nufft.h"
#include "toeplitz.h"
#include "types.h"

namespace tomocam {

    /**
     * Calculates the error between the model and the sinogram data
     *
     *  @param DeviceArray<T> Output from the gradient function
     *  @param DeviceArray<T> backprojection of the sinogram data
     *  @param cudaStream_t for concurrencny
     *  @return T error
     */ 
    template <typename T>
    T calc_error(DeviceArray<T> &, DeviceArray<T> &, cudaStream_t);

    /**
     * Computes back projection from sinograms using NUFFT
     *
     * @param DeviceArray<T> sinogram space
     * @param int center of rotation
     * @param NUFFT::Grid non-unifrom grid on which NUFFT is computed
     * @return DeviceArray<T> Image space
     */
    template <typename T>
    DeviceArray<T> backproject(const DeviceArray<T> &, const NUFFT::Grid<T> &,
        int, cudaStream_t);

    /**
     * Computes forward projection from a stack of images using NUFFT
     *
     * @param DeviceArray<T> Image space
     * @param int center of rotation
     * @param NUFFT::Grid non-unifrom grid on which NUFFT is computed
     * @return DeviceArray<T> sinogram space
     */
    template <typename T>
    DeviceArray<T> project(const DeviceArray<T> &, const NUFFT::Grid<T> &, int,
        cudaStream_t);

    /**
     * Parital calculation of the gradient of the objective function
     * \nabala f = R^*R\,x - R^*y + TV(f)
     * this function calculates the term R^*R\,x
     * the term R^*y is calculated at the beginning of the optimization
     *
     * @param DeviceArray<T> current solution
     * @param NUFFT::Grid non-unifrom grid on which NUFFT is computed
     * @param cudaStream_t CUDA stream for concurrencny
     */
    template <typename T>
    std::tuple<DeviceArray<T>, T> gradient(DeviceArray<T> &, DeviceArray<T> &,
        const NUFFT::Grid<T> &, cudaStream_t);

    /**
     * Parital calculation of the gradient of the objective function
     * \nabala f = R^*R\,x - R^*y + TV(f)
     * this function calculates the term R^*R\,x - R^*y, and,
     * (x R)^* R x - 2 R^*y.
     * y^*y is calculated at the beginning of the optimization
     */

    template <typename T>
    std::tuple<DeviceArray<T>, T> gradient2(DeviceArray<T> &, DeviceArray<T> &,
        const PointSpreadFunction<T> &, cudaStream_t);

} // namespace tomocam

#endif // TOMOCAM_INTERNALS__H
