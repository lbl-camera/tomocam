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
#include <fstream>

#include "dist_array.h"
#include "optimize.h"
#include "tomocam.h"
#include "machine.h"

namespace tomocam {

    template <typename T>
    void mbir(DArray<T> &sino,
        DArray<T> &model,
        float *angles,
        float center,
        int num_iters,
        float oversample,
        float sigma,
        float p) {

        Optimizer opt(
            model.dims(), sino.dims(), angles, center, oversample, sigma);

        std::ofstream fout("tomocam_mbir.log");
        model.init(1.f);
        for (int i = 0; i < num_iters; i++) {
            DArray<T> grad = model;
            gradient(grad, sino, angles, center, oversample);
            MachineConfig::getInstance().synchronize();
            float e = grad.norm();
            add_total_var(model, grad, p, sigma);
            opt.update(model, grad);
            fout << "Iteration:  " << i << ", Error: " << e << std::endl; 
        }
    }

    template void mbir<float>(DArray<float> &,
        DArray<float> &,
        float *,
        float,
        int,
        float,
        float,
        float);
} // namespace tomocam
