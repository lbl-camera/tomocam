#ifndef TOMOCAM_COVOLUTION__H
#define TOMOCAM_COVOLUTION__H

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "dist_array.h"
#include "types.h"

namespace tomocam {

    //TODO rescal
    void rescale(cuComplex_t *, dim3_t, float, cudaStream_t);

    // TODO document
    void deapodize(cuComplex_t *, dim3_t, float, float, cudaStream_t);

    // TODO document (find appropriate place for function declaration 
    void kaiser_window(kernel_t &, float, float, size_t, int);

    // TODO document
    void backProject(float *, float *, dim3_t, dim3_t, float, float, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void polarsample_transpose(cuComplex_t *, cuComplex_t *, dim3_t, dim3_t, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void forwardSim(float *, float *, dim3_t, dim3_t, float, float, DeviceArray<float>, kernel_t, cudaStream_t);

    // TODO document
    void polarsample(cuComplex_t *, cuComplex_t *, dim3_t, dim3_t, DeviceArray<float>, kernel_t, cudaStream_t);

} // namespace tomocam

#endif // TOMOCAM_COVOLUTION__H
