/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */


#include "dist_array.h"

#ifndef UTILS__H
#define UTILS__H

namespace tomocam {

    /**
     * @brief Zero pad the sinogram by a factor of \sqrt{2} and, 
     * shifts center of rotation to center of the sinogram
     *
     * @param sinogram
     * @param center of rotation
     *
     * @return zero padded sinogram
     */
    template <typename T>
    DArray<T> preproc(DArray<T> &, T);



    /**
     * @brief Crop the reconstruction to size provided
     *
     * @param reconstruction
     * @param number of entries in the cropped image
     *
     * @return cropped reconstruction
     */
    template <typename T>
    DArray<T> postproc(DArray<T> &, int);


} // namespace tomocam
#endif // UTILS__H
