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
#include <functional>
#include <pybind11/pybind11.h>

#include "dist_array.h"
#include "optimize.h"
#include "tomocam.h"
#include "machine.h"
#include "nufft.h"
#include "wrappers.h"

namespace tomocam {

    template <typename T>
    DArray<T> mbir(DArray<T> &sino, T *angles, T center, T sigma, T p, int num_iters, T step_size, T tol, T penalty) {

        // number of gpus available
        int ndevice = Machine::config.num_of_gpus();

        // recon dimensions
        int nproj = sino.dims().y;
        int nslcs = sino.dims().x;
        int ncols = sino.dims().z;

        dim3_t dims(nslcs, ncols, ncols);
        DArray<T> sinoT(dims);

        // back-project data
        back_project(sino, sinoT, angles, center);

        // sinogram dot sinogram
        T sino_norm = sino.norm();

        // create uniform-grid on each device
        std::cout << "starting to build NUGrids" << std::endl;
        NUFFTGrid *nugrids = new NUFFTGrid[ndevice];
        for (int dev_id = 0; dev_id < ndevice; dev_id++) {
            nugrids[dev_id] = NUFFTGrid(ncols, nproj, angles, dev_id); 
        }
        std::cout << "done building NUGrids" << std::endl;

        // setup wrapper for optimizer
        auto recon = MBIR<float>(&sinoT, nugrids, p, sigma);
        
        /* initialize solution vector */
        DArray<T> x0(dims);
        x0.init(static_cast<T>(1));

        auto calc_grad = std::bind(&MBIR<float>::grad, recon, std::placeholders::_1);
        auto calc_error = std::bind(&MBIR<float>::error, recon, std::placeholders::_1);
        auto pf = calc_grad(x0);

        /* divide step_size by Lipschitz */
        auto L = pf.max();
        step_size /= L; 
        std::cout << "step_size: " << step_size << ", Lipschitz: " << L << std::endl;

        return x0;
    }

    template DArray<float> mbir(
                                DArray<float>&, // sinogram
                                float *,  // angles 
                                float,    // center 
                                float,    // sigma
                                float,    // p 
                                int,      // num_iters
                                float,    // step_size
                                float,    // tol 
                                float     // penalty
                               );
                
} // namespace tomocam
