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
    DArray<T> mbir(DArray<T> &, float *, float, float, float, float, int);

    // specialize
    extern template DArray<float> mbir<float>(DArray<float> &, float *, float, float, float, float, int);

    /**
     * TODO: document it
     */
    void add_tv_hessian(DArray<float> &, float);

} // namespace tomocam

#endif // TOMOCAM__H
