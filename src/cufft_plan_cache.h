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
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, distribute copies to the public, prepare derivative
 * works, and perform publicly and display publicly, and to permit other to do
 * so.
 *---------------------------------------------------------------------------------
 */

#ifndef CUFFT_PLAN_CACHE__H
#define CUFFT_PLAN_CACHE__H

#include <mutex>
#include <optional>
#include <stdexcept>

#include "cufft_plan.h"
#include "types.h"

namespace tomocam::fft {

    template <typename T>
    class CufftPlanCache {
      private:
        struct PlanParams {
            dim3_t dims;

            bool operator==(const PlanParams &other) const {
                return dims.x == other.dims.x && dims.y == other.dims.y &&
                       dims.z == other.dims.z;
            }
        };

        CufftPlanWrapper<T> r2c_plan_;
        CufftPlanWrapper<T> c2r_plan_;
        std::once_flag r2c_init_flag_;
        std::once_flag c2r_init_flag_;
        std::optional<PlanParams> r2c_params_;
        std::optional<PlanParams> c2r_params_;

      public:
        CufftPlanCache() = default;
        ~CufftPlanCache() = default;

        CufftPlanCache(const CufftPlanCache &) = delete;
        CufftPlanCache &operator=(const CufftPlanCache &) = delete;
        CufftPlanCache(CufftPlanCache &&) = delete;
        CufftPlanCache &operator=(CufftPlanCache &&) = delete;

        CufftPlanWrapper<T> &get_plan(PlanType type, dim3_t dims) {
            PlanParams params{dims};

            if (type == PlanType::R2C) {
                std::call_once(r2c_init_flag_, [&]() {
                    r2c_plan_.make_plan(PlanType::R2C, dims);
                    r2c_params_ = params;
                });
                if (r2c_params_ && !(*r2c_params_ == params)) {
                    throw std::invalid_argument(
                        "R2C plan already cached with different parameters");
                }
                return r2c_plan_;
            } else if (type == PlanType::C2R) {
                std::call_once(c2r_init_flag_, [&]() {
                    c2r_plan_.make_plan(PlanType::C2R, dims);
                    c2r_params_ = params;
                });
                if (c2r_params_ && !(*c2r_params_ == params)) {
                    throw std::invalid_argument(
                        "C2R plan already cached with different parameters");
                }
                return c2r_plan_;
            } else {
                throw std::invalid_argument("Only R2C and C2R plans are supported");
            }
        }
    };

    namespace plans {
        template <typename T>
        inline CufftPlanCache<T> cache;
    }

} // namespace tomocam::fft

#endif // CUFFT_PLAN_CACHE__H
