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


#include "tomocam.h"

namespace tomocam {

    /*
     * ∇f = A'Ax - A'b  
     *  f = x'A'Ax - 2 x'A'b + b'b 
     */ 

    template <typename T>
    class MBIR {
        T xTATAx_;
        T bTb_;
        NUFFTGrid *nugrids_;
        T p_;
        T sigma_; 
        DArray<T> *bT_;


        public:
            MBIR(DArray<T> *sinoT, NUFFTGrid *nugrids, T p, T sigma):
                bT_(sinoT), nugrids_(nugrids), p_(p), sigma_(sigma) {}

            DArray<T> grad(DArray<T> x) {
                auto g = x;
                xTATAx_ = gradient(g, *bT_, nugrids_);
                add_total_var(x, g, p_, sigma_);
                return g;
            } 

            T error(DArray<T> x) {
                T xTATb = x.dot(*bT_);
                return (xTATAx_ -2*xTATb+ bTb_);
            }
    };

    template <typename T>
    inline T calc_lipschitz(DArray<T> &x, DArray<T> &sino,
        T *angles, T center, T oversample, T p, T sigma) {

        DArray<T> g(x.dims());
        DArray<T> y(sino.dims());
        radon(x, y, angles, center, oversample);
        radonT(y, g, angles, center, oversample);
        add_tv_hessian(g, sigma);
        return g.max();
    }
} // namespace tomocam
