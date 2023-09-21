#ifndef RESAMPLE__H
#define RESAMPLE__H

#include "dev_array.h"

namespace tomocam {
    template <typename T> 
    void upsample(DeviceArray<T> &, DeviceArray<T> &);

    template <typename T>
    void downsample(DeviceArray<T> &, DeviceArray<T> &, int);
} // namespace
#endif 
