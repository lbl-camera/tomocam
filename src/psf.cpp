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

#include <iostream>

#include "machine.h"
#include "dev_array.h"
#include "internals.h"
#include "types.h"

namespace tomocam {

    DeviceArray<float> calc_psf(NUFFTGrid &grid, int nrows) {

        int  M = grid.size();
        int N1 = 2 * nrows;
        dim3_t dims(1, N1, N1);

        // allocate ones
        auto ones = DeviceArray<cuComplex_t>(dims);
        const a = cuComplex_t(1.f, 0);
        ones.init(a);

        // allocate psf
        auto  psf = DeviceArray<cuComplex_t>(dims);
        
        cufinufftf_plan plan;
        NUFFT_CALL(nufftPlan1(dims, grid, plan));
        NUFFT_CALL(cufinufftf_execute(ones.dev_ptr(), psf.dev_ptr(), plan));
        NUFFT_CALL(cufinufftf_destroy(plan));    
        return cmplx_to_real(psf, cudaStreamPerThread);
    }
}
