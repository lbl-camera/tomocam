/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 *Software to reproduce, distribute copies to the public, prepare derivative
 *works, and perform publicly and display publicly, and to permit other to do
 *so.
 *---------------------------------------------------------------------------------
 */

#include <iostream>
#include <utility>
#include <pybind11/pybind11.h>

#include "dist_array.h"
#include "optimize.h"
#include "tomocam.h"
#include "machine.h"

namespace tomocam {

    template <typename T>
    DArray<T> mbir(DArray<T> &sino,
        float *angles,
        float center,
        float oversample,
        float sigma,
        float p,
        int num_iters) {

        dim3_t dims = sino.dims();
        dims.y = dims.z;

        // normalize
        T minv = sino.min();
        T maxv = sino.max();
        sino = (sino - minv) / (maxv - minv);

        Optimizer<T> opt(dims, num_iters);
        return opt.minimize(sino, angles, center, oversample, p, sigma);
    }

    template DArray<float> mbir<float>(DArray<float> &, 
        float *, float, float, float, float, int);
} // namespace tomocam
