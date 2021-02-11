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

#include <cmath>

#include "dist_array.h"
#include "tomocam.h"

#ifndef TOMOCAM_OPTIMIZE__H
#define TOMOCAM_OPTIMIZE__H
namespace tomocam {
    class Optimizer {
      private:
        float lipschitz_;
        DArray<float> t_;
        DArray<float> z_;

      public:
        Optimizer(dim3_t recon_dims,
            dim3_t sino_dims,
            float *angles,
            float cen,
            float over_samp,
            float sigma) :
            t_(recon_dims), z_(recon_dims) {

            dim3_t d1 = recon_dims;
            d1.x = 1;

            dim3_t d2 = sino_dims;
            d2.x = 1;

            DArray<float> x(d1);
            x.init(1);
            DArray<float> g(d1);
            DArray<float> y(d2);

            radon(x, y, angles, cen, over_samp);
            iradon(y, g, angles, cen, over_samp);
            add_tv_hessian(g, sigma);
            lipschitz_ = 1.0E-02 / g.max();
        }

        template <typename T>
        void update(DArray<T> &recon, DArray<T> &gradient) {
#pragma omp parallel for
            for (int i = 0; i < recon.size(); i++) {
                float z_new = recon[i] - lipschitz_ * gradient[i];
                float t_new = 0.5 * (1 + std::sqrt(1 + 4 * t_[i] * t_[i]));
                recon[i] = z_new + (t_[i] - 1) / t_new * (z_new - z_[i]);
                t_[i] = t_new;
                z_[i] = z_new;
            }
        }
    };
} // namespace tomocam

#endif // TOMOCAM_OPTIMIZE__H
