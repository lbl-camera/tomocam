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
#include "types.h"
#include "nufft.h"

namespace tomocam {

    /**
     * Calculates differene between model and data
     *
     *  @param DeviceArray<cuComplex_t> model as complex array on device
     *  @param DeviceArray<float> sinogram data as array on device
     *  @param int size of padding
     *  @param cudaStream_t for concurrencny
     */ 
    void calc_error(dev_arrayZ &, dev_arrayF &, cudaStream_t);

    /**
     * Rescales output from cufft, by dividing by N^2
     *
     * @param DeviceArray<cuComplex_t> cufft output
     * @param float scalar
     * @param cudaStream_t for concurrencny
     */ 
    void rescale(dev_arrayZ &, float, cudaStream_t);


    /**
     * Computes back projection from sinograms using NUFFT
     *
     * @param DeviceArray<cuComplex_t> sinogram
     * @param DeviceArray<cuComplex_t> image space
     * @param float correction to the center of rotation
     * @param NUFFTGrid non-unifrom grid on which NUFFT is computed
     */ 
    void back_project(dev_arrayZ &, dev_arrayZ &, float, NUFFTGrid &);

    /**
     * Computes forward projection from a stack of images using NUFFT
     *
     * @param DeviceArray<cuComplex_t> Image stack
     * @param DeviceArray<cuComplex_t> Computed sinograms
     * @param float correction to the center of rotation
     * @param NUFFTGrid non-unifrom grid on which NUFFT is computed
     */ 
    void project(dev_arrayZ &, dev_arrayZ &, float, NUFFTGrid &);


    /**
     * Calculates the gradients, in-place
     *
     * @param DeviceArray<cuComplex_t> Model
     * @param DeviceArray<float> Data
     * @param float Center correction (+ padding/2)
     * @param DeviceArray<float> Projection angles
     * @param kernel_t Window function for convolution
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void calc_gradient(dev_arrayZ &, dev_arrayF &, float, NUFFTGrid &);

    /**
     * Calculates constrains on the objective function, and updates gradients in-place
     *
     * @param DeviceArray<float> Gradients of obj. function
     * @param DeviceArray<float> Model
     * @param float Surrogate model paramter
     * @param float Surrogate model paramter
     * @param cudaStream_t for concurrencny
     */ 
    void add_total_var(dev_arrayF &, dev_arrayF &, float, float, cudaStream_t);

    /**
     * Typecast from float to cuComplex_t
     *
     * @param DeviceArray<float> input
     * @param cudaStream_t cuda stream for concurrencny
     */
    dev_arrayZ real_to_cmplx(dev_arrayF &, cudaStream_t);

    /**
     * Typecast from cuComplex_t to float
     *
     * @param DeviceArray<cuComplex_t> input
     * @param cudaStream_t cuda stream for concurrencny
     */
    dev_arrayF cmplx_to_real(dev_arrayZ &, cudaStream_t);

} // namespace tomocam

#endif // TOMOCAM_INTERNALS__H
