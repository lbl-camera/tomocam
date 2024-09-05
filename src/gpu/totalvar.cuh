
#include <cuda_runtime.h>
#include <cuda.h>

#include "dev_array.h"

#ifndef TOTALVAR__CUH
#define TOTALVAR__CUH

namespace tomocam {
    namespace gpu {

        /** @brief Compute the gradient of total variation of an image, and
         update the the current gradient /f$ \nabla f = \nabla f + p \nabla
         g(|f|; p, \sigma) /f$
         *
         * @param[in] img The current solution
         * @param[inout] grad The gradient of the current solution
         * @param[in] p The weight of the total variation term
         * @param[in] /f$\sigma/f$ The weight of the data term
         */
        template <typename T>
        void add_total_var(const DeviceArray<T> &, DeviceArray<T> &, float,
            float);

    } // namespace gpu
} // namespace tomocam
#endif // TOTALVAR__CUH
