/* -------------------------------------------------------------------------------
* Tomocam Copyright (c) 2018
*
* The Regents of the University of California, through Lawrence Berkeley National
* Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
*  Energy). All rights reserved.
*
* If you have questions about your rights to use or distribute this software,
* please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
*
* NOTICE. This Software was developed under funding from the U.S. Department of
* Energy and the U.S. Government consequently retains certain rights. As such, the
* U.S. Government has been granted for itself and others acting on its behalf a
* paid-up, nonexclusive, irrevocable, worldwide license in the Software to
* reproduce, distribute copies to the public, prepare derivative works, and
* perform publicly and display publicly, and to permit other to do so.
*---------------------------------------------------------------------------------
*/

#ifndef TOMOCAM_FFT__H
#define TOMOCAM_FFT__H

#include <cufft.h>

#include "dist_array.h"

namespace tomocam {
    template <class T1, class T2>
    void fft1d(DArray<T1> &, DArray<T2> &);

    template <class T1, class T2>
    void fft2d(DArray<T1> &, DArray<T2> &);

    template <class T1, class T2>
    void ifft1d(DArray<T1> &, DArray<T2> &);

    template <class T1, class T2>
    void ifft2d(DArray<T1> &, DArray<T2> &);
} // namespace tomocam
#endif // TOMOCAM_FFT__H
