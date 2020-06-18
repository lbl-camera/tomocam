#ifndef TOMOCAM__H
#define TOMOCAM__H

#include "dist_array.h"

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
    void axpy(float, DArray<float> &, DArray<float> &);

    // TODO document
    float norm2(DArray<float> &);
} // namespace tomocam

#endif // TOMOCAM__H
