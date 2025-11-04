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
#include <vector>
#include <string>

#include "dist_array.h"
#include "tomocam.h"
#include "hdf5/writer.h"

#ifdef MULTIPROC
#include "multiproc.h"
#endif

namespace tomocam {

template<typename T>
DArray<T> fbp_helper(DArray<T> &sino, std::vector<T> &angs, int center) {
    int nrays = sino.ncols();
    auto sino2 = preproc(sino, static_cast<T>(center));
    auto recon = backproject(sino2, angs, true);
    return postproc(recon, nrays);
}

template<typename T>
DArray<T> mbir_mpi(
    DArray<T> &sino,
    std::vector<T> &angles,
    T center,
    int num_iters,
    T sigma,
    T tol,
    T xtol,
    bool file_write,
    const std::string &output_file) {

#ifdef MULTIPROC
    int nprocs = multiproc::mp.nprocs();
    int myrank = multiproc::mp.myrank();
#else
    int nprocs = 1;
    int myrank = 0;
#endif

    if (sino.ncols() % 2 == 0) {
        sino.dropcol();
    }

    auto x0 = fbp_helper(sino, angles, static_cast<int>(center));
    x0.normalize();

    auto recon = mbir(x0, sino, angles, center, num_iters, sigma, tol, xtol);

#ifdef MULTIPROC
    int nrows = sino.nrows();
    int ncols = sino.ncols();
    int local_nslcs = recon.nslices();
    
    std::vector<int> all_nslcs(nprocs);
    multiproc::mp.Gather(&local_nslcs, 1, all_nslcs.data(), 1, 0);
    
    int total_nslcs = 0;
    std::vector<int> displs(nprocs, 0);
    std::vector<int> recvcounts(nprocs, 0);
    
    if (myrank == 0) {
        for (int i = 0; i < nprocs; i++) {
            total_nslcs += all_nslcs[i];
            recvcounts[i] = all_nslcs[i] * nrows * ncols;
            if (i > 0) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
    }
    
    DArray<T> combined_recon({0, 0, 0});
    if (myrank == 0) {
        combined_recon = DArray<T>({total_nslcs, nrows, ncols});
    }
    
    multiproc::mp.Gatherv(recon.begin(), recon.size(),
                          myrank == 0 ? combined_recon.begin() : nullptr,
                          recvcounts.data(), displs.data(), 0);
    
    if (file_write && myrank == 0 && !output_file.empty()) {
        h5::Writer writer(output_file.c_str());
        writer.write("recon", combined_recon);
    }
    
    if (myrank == 0) {
        return combined_recon;
    } else {
        return DArray<T>({0, 0, 0});
    }
#else
    if (file_write && !output_file.empty()) {
        h5::Writer writer(output_file.c_str());
        writer.write("recon", recon);
    }
    
    return recon;
#endif
}

template DArray<float> mbir_mpi(DArray<float> &, std::vector<float> &, float, int, float, float, float, bool, const std::string &);
template DArray<double> mbir_mpi(DArray<double> &, std::vector<double> &, double, int, double, double, double, bool, const std::string &);

} // namespace tomocam
