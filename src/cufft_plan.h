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

#ifndef CUFFT_PLAN__H
#define CUFFT_PLAN__H

#include <cufft.h>
#include <stdexcept>

#include "types.h"

namespace tomocam::fft {

    template <typename T>
    struct CufftTraits;

    template <>
    struct CufftTraits<double> {
        using plan_type = cufftHandle;
        static constexpr cufftType r2c_type = CUFFT_D2Z;
        static constexpr cufftType c2r_type = CUFFT_Z2D;

        static int makeplan_r2c(dim3_t dims, plan_type *plan) {
            int rank = 2;
            int n[] = {static_cast<int>(dims.y), static_cast<int>(dims.z)};
            int istride = 1;
            int ostride = 1;
            int idist = dims.y * dims.z;
            int odist = dims.y * (dims.z / 2 + 1);
            int batches = dims.x;

            return cufftPlanMany(plan, rank, n, NULL, istride, idist, NULL, ostride,
                                 odist, r2c_type, batches);
        }

        static int makeplan_c2r(dim3_t dims, plan_type *plan) {
            int rank = 2;
            int n[] = {static_cast<int>(dims.y), static_cast<int>(dims.y)};
            int istride = 1;
            int ostride = 1;
            int idist = dims.y * dims.z;
            int odist = dims.y * dims.y;
            int batches = dims.x;

            return cufftPlanMany(plan, rank, n, NULL, istride, idist, NULL, ostride,
                                 odist, c2r_type, batches);
        }

        static int rfft2d(plan_type plan, double *in, cufftDoubleComplex *out) {
            return cufftExecD2Z(plan, in, out);
        }

        static int irfft2d(plan_type plan, cufftDoubleComplex *in, double *out) {
            return cufftExecZ2D(plan, in, out);
        }

        static void destroy(plan_type plan) { cufftDestroy(plan); }
    };

    template <>
    struct CufftTraits<float> {
        using plan_type = cufftHandle;
        static constexpr cufftType r2c_type = CUFFT_R2C;
        static constexpr cufftType c2r_type = CUFFT_C2R;

        static int makeplan_r2c(dim3_t dims, plan_type *plan) {
            int rank = 2;
            int n[] = {static_cast<int>(dims.y), static_cast<int>(dims.z)};
            int istride = 1;
            int ostride = 1;
            int idist = dims.y * dims.z;
            int odist = dims.y * (dims.z / 2 + 1);
            int batches = dims.x;

            return cufftPlanMany(plan, rank, n, NULL, istride, idist, NULL, ostride,
                                 odist, r2c_type, batches);
        }

        static int makeplan_c2r(dim3_t dims, plan_type *plan) {
            int rank = 2;
            int n[] = {static_cast<int>(dims.y), static_cast<int>(dims.y)};
            int istride = 1;
            int ostride = 1;
            int idist = dims.y * dims.z;
            int odist = dims.y * dims.y;
            int batches = dims.x;

            return cufftPlanMany(plan, rank, n, NULL, istride, idist, NULL, ostride,
                                 odist, c2r_type, batches);
        }

        static int rfft2d(plan_type plan, float *in, cufftComplex *out) {
            return cufftExecR2C(plan, in, out);
        }
        static int irfft2d(plan_type plan, cufftComplex *in, float *out) {
            return cufftExecC2R(plan, in, out);
        }

        static void destroy(plan_type plan) { cufftDestroy(plan); }
    };

    enum class PlanType { R2C, C2R };

    template <typename T>
    class CufftPlanWrapper {
      private:
        using Traits = CufftTraits<T>;
        typename Traits::plan_type plan;
        bool initialized = false;

      public:
        CufftPlanWrapper() = default;

        void make_plan(PlanType type, dim3_t dims) {
            int ierr;
            if (type == PlanType::R2C) {
                ierr = Traits::makeplan_r2c(dims, &plan);
            } else {
                ierr = Traits::makeplan_c2r(dims, &plan);
            }

            if (ierr != CUFFT_SUCCESS) {
                throw std::runtime_error("Error in cufftPlanMany");
            }
            initialized = true;
        }

        void rfft2d(T *in, cuda::std::complex<T> *out) {
            if (!initialized) {
                throw std::runtime_error(
                    "CufftPlanWrapper::rfft2d called before make_plan");
            }
            int ierr = Traits::rfft2d(plan, in, out);
            if (ierr != CUFFT_SUCCESS) {
                throw std::runtime_error("Error in cufftExecR2C");
            }
        }

        void irfft2d(cuda::std::complex<T> *in, T *out) {
            if (!initialized) {
                throw std::runtime_error(
                    "CufftPlanWrapper::irfft2d called before make_plan");
            }
            int ierr = Traits::irfft2d(plan, in, out);
            if (ierr != CUFFT_SUCCESS) {
                throw std::runtime_error("Error in cufftExecC2R");
            }
        }

        cufftHandle get() const {
            if (!initialized) {
                throw std::runtime_error(
                    "CufftPlanWrapper::get_plan called before make_plan");
            }
            return plan;
        }

        ~CufftPlanWrapper() {
            if (initialized) { Traits::destroy(plan); }
        }

        CufftPlanWrapper(const CufftPlanWrapper &) = delete;
        CufftPlanWrapper &operator=(const CufftPlanWrapper &) = delete;

        bool valid() const { return initialized; }

        CufftPlanWrapper(CufftPlanWrapper &&other) noexcept
            : plan(other.plan), initialized(other.initialized) {
            other.initialized = false;
        }

        CufftPlanWrapper &operator=(CufftPlanWrapper &&other) noexcept {
            if (this != &other) {
                if (initialized) { Traits::destroy(plan); }
                plan = other.plan;
                initialized = other.initialized;
                other.initialized = false;
            }
            return *this;
        }
    };

} // namespace tomocam::fft

#endif // CUFFT_PLAN__H
