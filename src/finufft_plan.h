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

#ifndef FINUFFT_PLAN__H
#define FINUFFT_PLAN__H

#include <array>
#include <cuda/std/complex>
#include <cufinufft.h>

namespace tomocam::nufft {

    template <typename T>
    struct FinufftTraits;

    template <>
    struct FinufftTraits<double> {
        using plan_type = cufinufft_plan;
        using complex_type = cuda::std::complex<double>;

        static int makeplan(int type, int dim, std::array<int64_t, 2> n_modes,
                            int iflag, int ntrans, plan_type *plan,
                            cufinufft_opts *opts) {

            double tol = 1.0e-14;
            return cufinufft_makeplan(type, dim, n_modes.data(), iflag, ntrans, tol,
                                      plan, opts);
        }

        static int setpts(plan_type plan, int64_t npts, double *x, double *y,
                          double *z, int nk, double *s, double *t, double *u) {
            return cufinufft_setpts(plan, npts, x, y, z, nk, s, t, u);
        }

        static int execute(plan_type plan, complex_type *cz, complex_type *fz) {
            // cast to cuDoubleComplex
            cuDoubleComplex *cptr = reinterpret_cast<cuDoubleComplex *>(cz);
            cuDoubleComplex *fptr = reinterpret_cast<cuDoubleComplex *>(fz);
            return cufinufft_execute(plan, cptr, fptr);
        }

        static void destroy(plan_type plan) { cufinufft_destroy(plan); }
    };

    template <>
    struct FinufftTraits<float> {
        using plan_type = cufinufftf_plan;
        using complex_type = cuda::std::complex<float>;

        static int makeplan(int type, int dim, std::array<int64_t, 2> n_modes,
                            int iflag, int ntrans, plan_type *plan,
                            cufinufft_opts *opts) {
            float tol = 1.2e-6f;
            return cufinufftf_makeplan(type, dim, n_modes.data(), iflag, ntrans, tol,
                                       plan, opts);
        }

        static int setpts(plan_type plan, int64_t npts, float *x, float *y, float *z,
                          int nk, float *s, float *t, float *u) {
            return cufinufftf_setpts(plan, npts, x, y, z, nk, s, t, u);
        }

        static int execute(plan_type plan, complex_type *cz, complex_type *fz) {
            // cast to cuFloatComplex
            cuFloatComplex *cptr = reinterpret_cast<cuFloatComplex *>(cz);
            cuFloatComplex *fptr = reinterpret_cast<cuFloatComplex *>(fz);
            return cufinufftf_execute(plan, cptr, fptr);
        }

        static void destroy(plan_type plan) { cufinufftf_destroy(plan); }
    };

    template <typename T>
    class FinufftPlanWrapper {
      private:
        using Traits = FinufftTraits<T>;
        typename Traits::plan_type plan;
        bool initialized = false;

      public:
        FinufftPlanWrapper() = default;

        void make_plan(int type, int dim, std::array<int64_t, 2> n_modes, int iflag,
                       int device_id) {
            cufinufft_opts opts;
            cufinufft_default_opts(&opts);
            opts.upsampfac = 2.0;
            opts.gpu_device_id = device_id;
            int ierr = Traits::makeplan(type, dim, n_modes, iflag, 1, &plan, &opts);
            if (ierr != 0) {
                throw std::runtime_error("Error in cufinufft_makeplan");
            }
            initialized = true;
        }

        void set_points(int64_t npts, T *x, T *y) {
            if (!initialized) {
                throw std::runtime_error(
                    "FinufftPlanWrapper::set_points called before make_plan");
            }
            int ierr = Traits::setpts(plan, npts, x, y, nullptr, 0, nullptr, nullptr,
                                      nullptr);
            if (ierr != 0) { throw std::runtime_error("Error in cufinufft_setpts"); }
        }

        int execute(cuda::std::complex<T> *cz, cuda::std::complex<T> *fz) {
            if (!initialized) {
                throw std::runtime_error(
                    "FinufftPlanWrapper::execute called before make_plan");
            }
            return Traits::execute(plan, (typename Traits::complex_type *)cz,
                                   (typename Traits::complex_type *)fz);
        }

        ~FinufftPlanWrapper() {
            if (initialized) { Traits::destroy(plan); }
        }

        FinufftPlanWrapper(const FinufftPlanWrapper &) = delete;
        FinufftPlanWrapper &operator=(const FinufftPlanWrapper &) = delete;

        bool valid() const { return initialized; }

        FinufftPlanWrapper(FinufftPlanWrapper &&other) noexcept
            : plan(other.plan), initialized(other.initialized) {
            other.initialized = false;
        }

        FinufftPlanWrapper &operator=(FinufftPlanWrapper &&other) noexcept {
            if (this != &other) {
                if (initialized) { Traits::destroy(plan); }
                plan = other.plan;
                initialized = other.initialized;
                other.initialized = false;
            }
            return *this;
        }
    };

} // namespace tomocam::nufft

#endif // FINUFFT_PLAN__H
