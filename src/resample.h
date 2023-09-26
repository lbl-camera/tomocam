#ifndef RESAMPLE__H
#define RESAMPLE__H

#include "dev_array.h"

namespace tomocam {
    void upsample(DeviceArray<float> &, DeviceArray<float> &);

    void downsample(DeviceArray<float> &, DeviceArray<float> &, int);
} // namespace
#endif 
