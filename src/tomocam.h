#ifndef TOMOCAM_COVOLUTION__H
#define TOMOCAM_COVOLUTION__H

#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "dist_array.h"
#include "types.h"

namespace tomocam {

    // TODO document
    void iradon(DArray<float> &, DArray<float> &, float *, float, float);


    // TODO document
    void radon(DArray<float> &, DArray<float> &, float *, float, float);


} // namespace tomocam

#endif // TOMOCAM_COVOLUTION__H
