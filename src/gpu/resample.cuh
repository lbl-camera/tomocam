
#include "dev_array.h"

#ifndef RESAMPLE__CUH
#define RESAMPLE__CUH

namespace tomocam::gpu {
    template <typename T>
    DeviceArray<T> lanczos_upsampling(const DeviceArray<T> &, dim3_t dims);
} // namespace tomocam::gpu

#endif // RESAMPLE__CUH
