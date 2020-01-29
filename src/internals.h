#ifndef TOMOCAM_INTERNALS__H
#define TOMOCAM_INTERNALS__H

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "kernel.h"
#include "dist_array.h"
#include "types.h"

namespace tomocam {

    // TODO document
    void calc_error(cuComplex_t *, float *, dim3_t , dim3_t, cudaStream_t);

    //TODO rescale
    void rescale(cuComplex_t *, dim3_t, float, cudaStream_t);

    // TODO document
    void deapodize(cuComplex_t *, dim3_t, float, float, cudaStream_t);

    // TODO document (find appropriate place for function declaration 
    void kaiser_window(kernel_t &, float, float, size_t, int);

    // TODO document
    void back_project(cuComplex_t *, cuComplex_t *, dim3_t, dim3_t, float, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void stage_back_project(float *, float *, dim3_t, dim3_t, float, float, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void polarsample_transpose(cuComplex_t *, cuComplex_t *, dim3_t, dim3_t, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void fwd_project(cuComplex_t *, cuComplex_t *, dim3_t, dim3_t, float, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void stage_fwd_project(float *, float *, dim3_t, dim3_t, float, float, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void polarsample(cuComplex_t *, cuComplex_t *, dim3_t, dim3_t, DeviceArray<float>, kernel_t, cudaStream_t);

} // namespace tomocam

#endif // TOMOCAM_INTERNALS__H
