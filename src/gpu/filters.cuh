

#include "dev_array.h"
#include "types.h"

#ifndef CPU_FILTERS__CUH
#define CPU_FILTERS__CUH

namespace tomocam {
    namespace gpu {

        template <typename T>
        void apply_filter(DeviceArray<gpu::complex_t<T>> &);

    } // namespace gpu
}      // namespace tomocam
#endif // CPU_FILTERS__CUH
