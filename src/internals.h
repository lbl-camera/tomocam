#ifndef TOMOCAM_INTERNALS__H
#define TOMOCAM_INTERNALS__H

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "kernel.h"
#include "dist_array.h"
#include "types.h"

namespace tomocam {

    /**
     * Calculates differene between model and data, and resets zero-padding in model
     *
     *  @param DeviceArray<cuComplex_t> model as complex array on device
     *  @param DeviceArray<float> sinogram data as array on device
     *  @param cudaStream_t for concurrencny
     */ 
    void calc_error(dev_arrayc, dev_arrayf, cudaStream_t);

    /**
     * Rescales output from cufft, by dividing by N^2
     *
     * @param DeviceArray<cuComplex_t> cufft output
     * @param cudaStream_t for concurrencny
     */ 
    void rescale(dev_arrayc, cudaStream_t);

    /**
     * Deconvolves the NUFFT output with the convolution kernel for forward projection
     *
     * @param DeviceArray<cuComplex_t> NUFFT output
     * @param kernel_t convolution kernel (Kaiser window)
     * @param cudaStream_t for concurrencny
     */ 
    void deapodize1D(dev_arrayc, kernel_t, cudaStream_t);

    /**
     * Deconvolves the NUFFT output with the convolution kernel for backward projection
     *
     * @param DeviceArray<cuComplex_t> NUFFT output
     * @param kernel_t convolution kernel (Kaiser window)
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void deapodize2D(dev_arrayc, kernel_t, cudaStream_t);

    /**
     * Computes back projection from sinograms using NUFFT
     *
     * @param DeviceArray<cuComplex_t> sinogram
     * @param DeviceArray<cuComplex_t> reconstructed volume
     * @param float correction to the center of rotation
     * @param DeviceArray<float> angles at the projections
     * @param kernel_t convolution kernel (Kaiser window)
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void back_project(dev_arrayc, dev_arrayc, float, dev_arrayf, kernel_t, cudaStream_t);

    /**
     * Rconstructs voxels from sinograms (inverse radon transform)
     *
     * @param DeviceArray<cuComplex_t> partial sinogram data on device
     * @param DeviceArray<cuComplex_t> reconstructed output, corresponding to input sinogram
     * @param int Padding for oversampling the FFT
     * @param float Correction to the center of rotation (correction + padding/2)
     * @param DeviceArray<float> Projection angles
     * @param kernel_t Window function for convolution
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void stage_back_project(dev_arrayc, dev_arrayc &, int, float, dev_arrayf, kernel_t, cudaStream_t);

    /**
     * Wrapper to launch CUDA kernel for computing covolutions (Polar -> Cartesian)
     *
     * @param DeviceArray<cuComplex_t> Sinograms in the Fourier space on a polar grid
     * @param DeviceArray<cuComplex_t> Output
     * @param DeviceArray<float> Projection angles
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void polarsample_transpose(dev_arrayc , dev_arrayc, dev_arrayf, kernel_t, cudaStream_t);

    /**
     * Computes forward projection from voxels using NUFFT
     *
     * @param DeviceArray<cuComplex_t> Voxels
     * @param DeviceArray<cuComplex_t> Computed sinograms
     * @param float correction to the center of rotation
     * @param DeviceArray<float> angles at the projections
     * @param kernel_t convolution kernel (Kaiser window)
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void fwd_project(dev_arrayc, dev_arrayc, float, dev_arrayf, kernel_t, cudaStream_t);

    /**
     * Computes projections (sinograms) from voxels (Radon transform)
     *
     * @param DeviceArray<cuComplex_t> Partial voxel data on GPU memory
     * @param DeviceArray<cuComplex_t> Rconstructed sinograms corresponding to input voxels
     * @param int Padding for oversampling the FFT
     * @param float Correction to the center of rotation (correction + padding/2)
     * @param DeviceArray<float> Projection angles
     * @param kernel_t Window function for convolution
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void stage_fwd_project(dev_arrayc, dev_arrayc &, int, float, dev_arrayf, kernel_t, cudaStream_t);

    /**
     * Wrapper to launch CUDA kernel for computing covolutions (Cartesian -> Polar)
     *
     * @param DeviceArray<cuComplex_t> Oversampled FFT of voxles on a Cartesian grid
     * @param DeviceArray<cuComplex_t> Output of covolution  on a polar-grid
     * @param DeviceArray<float> Projection angles
     * @param cudaStream_t CUDA stream for concurrencny
     */ 
    void polarsample(dev_arrayc, dev_arrayc, dev_arrayf, kernel_t, cudaStream_t);

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
    void calc_gradient(dev_arrayc &, dev_arrayf, int, float, dev_arrayf, kernel_t, cudaStream_t);

    /**
     * Calculates constrains on the objective function, and updates gradients in-place
     *
     * @param DeviceArray<float> Gradients of obj. function
     * @param DeviceArray<float> Model
     * @param float Surrogate model paramter
     * @param float Surrogate model paramter
     * @param cudaStream_t for concurrencny
     */ 
    void add_total_var(dev_arrayf, dev_arrayf, float, float, cudaStream_t);

} // namespace tomocam

#endif // TOMOCAM_INTERNALS__H
