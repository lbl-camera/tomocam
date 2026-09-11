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

#ifndef FINITDIFF__CUH
#define FINITDIFF__CUH

namespace tomocam {
    namespace gpu {

        /** @brief forward-difference gradient of a 3D array:
         *  dudx[i] = u(i+1,j,k) - u(i,j,k), dudy/dudz analogous along y/z.
         *  Boundary handled via DeviceMemory<T>::at()'s clamped (Neumann)
         *  accessor. Output arrays must already be allocated to u's
         *  (halo-free) size.
         */
        template <typename T>
        void grad_u(const DeviceArray<T> &u, DeviceArray<T> &dudx,
                    DeviceArray<T> &dudy, DeviceArray<T> &dudz);

        /** @brief backward-difference divergence of a 3-component vector
         *  field (p, q, r) -- the exact discrete adjoint of grad_u:
         *  div[i] = (p(i)-p(i-1)) + (q(i)-q(i-1)) + (r(i)-r(i-1))
         */
        template <typename T>
        void divergence(const DeviceArray<T> &p, const DeviceArray<T> &q,
                        const DeviceArray<T> &r, DeviceArray<T> &div);

    } // namespace gpu
} // namespace tomocam
#endif // FINITDIFF__CUH
