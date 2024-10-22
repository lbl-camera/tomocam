/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

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
