#ifndef TOMOCAM__H
#define TOMOCAM__H

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

    // TODO document
    void gradient(DArray<float> &, DArray<float> &, float *, float, float);

    // TODO document
    void add_total_var(DArray<float> &, DArray<float> &, float , float);

    // TODO document
    float lipschitz();

} // namespace tomocam

#endif // TOMOCAM__H
