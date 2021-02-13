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
    void add_total_var(DArray<float> &, DArray<float> &, float, float);

    // TODO document
    void axpy(float, DArray<float> &, DArray<float> &);

	template <typename T>
    void mbir(DArray<T> &, DArray<T> &, float *, float, int, float, float, float);

    // specialize
    extern template void mbir<float>(
        DArray<float> &, DArray<float> &, float *, float, int, float, float, float);

    /**
     * TODO: document it
     */
    void add_tv_hessian(DArray<float> &, float);

} // namespace tomocam

#endif // TOMOCAM__H
