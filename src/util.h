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

#ifndef TOMOCAM_UTIL__H
#define TOMOCAM_UTIL__H

#include <vector>

namespace tomocam {
    std::vector<int> distribute(int work, int workers) {
        std::vector<int> share;
        int work_per_worker = work / workers;
        int extra = work % workers;
        for (int i = 0; i < workers; i++)
            if (i < extra ) share.push_back(work_per_worker+1);
            else share.push_back(work_per_worker);
        return share;
    }

}
#endif // TOMOCAM_UTIL__H
