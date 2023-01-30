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
     * Calculates differene between model and data, and resets zero-padding in model
     *
     *  @param DeviceArray<cuComplex_t> model as complex array on device
     *  @param DeviceArray<float> sinogram data as array on device
     *  @param int size of padding
     *  @param cudaStream_t for concurrencny
     */ 
    void calc_error(dev_arrayc &, dev_arrayf &, int ipad, cudaStream_t);

    /**
     * Rescales output from cufft, by dividing by N^2
     *
     * @param DeviceArray<cuComplex_t> cufft output
     * @param float scalar
     * @param cudaStream_t for concurrencny
     */ 
    void rescale(dev_arrayc &, float, cudaStream_t);


    /**
     * Computes back projection from sinograms using NUFFT
     *
     * @param DeviceArray<cuComplex_t> sinogram
     * @param DeviceArray<cuComplex_t> image space
     * @param float correction to the center of rotation
     * @param NUFFTGrid non-unifrom grid on which NUFFT is computed
     */ 
    void back_project(dev_arrayc &, dev_arrayc &, float, NUFFTGrid &);

    /**
     * Computes forward projection from a stack of images using NUFFT
     *
     * @param DeviceArray<cuComplex_t> Image stack
     * @param DeviceArray<cuComplex_t> Computed sinograms
     * @param float correction to the center of rotation
     * @param NUFFTGrid non-unifrom grid on which NUFFT is computed
     */ 
    void project(dev_arrayc &, dev_arrayc &, float, NUFFTGrid &);


    /**
     * Calculates the gradients, in-place
     *
     * @param DeviceArray<cuComplex_t> Model
     * @param DeviceArray<float> Data
     * @param int Padding for oversampling the FFT
     * @param float Center correction (+ padding/2)
     * @param DeviceArray<float> Projection angles
     * @param kernel_t Window function for convolution
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void calc_gradient(dev_arrayc &, dev_arrayf &, int, float, NUFFTGrid &);

    /**
     * Calculates gradient of constraint on the objective function, and updates gradients in-place
     *
     * @param DeviceArray<float> Model
     * @param DeviceArray<float> Gradients of obj. function
     * @param float Surrogate model paramter p
     * @param float Surrogate model paramter sigma
     * @param float Surrogate model paramter lambda
     * @param cudaStream_t for concurrencny
     */ 
    void add_total_var(dev_arrayf &, dev_arrayf &, float, float, float, cudaStream_t);

    /**
     * Calculates constraint on the objective function, and adds to objective function
     *
     * @param DeviceArray<float> Model
     * @param DeviceArray<float> Obj. function
     * @param float Surrogate model paramter p
     * @param float Surrogate model paramter sigma
     * @param float Surrogate model paramter lambda
     * @param cudaStream_t for concurrencny
     */
    void add_tv_func(dev_arrayf &, dev_arrayf &, float, float, float, cudaStream_t);

    /**
     * Adds zero padding and typecast from float to cuComplex_t
     *
     * @param DeviceArray<float> input
     * @param int3 padding
     * @param cudaStream_t cuda stream for concurrencny
     */
    dev_arrayc add_paddingR2C(dev_arrayf &, int3, cudaStream_t);

    /**
     * Removes zero padding and typecast from cuComplex_t to float
     *
     * @param DeviceArray<cuComplex_t> input
     * @param int3 padding
     * @param cudaStream_t cuda stream for concurrencny
     */
    dev_arrayf remove_paddingC2R(dev_arrayc &, int3, cudaStream_t);

} // namespace tomocam

#endif // TOMOCAM_INTERNALS__H
