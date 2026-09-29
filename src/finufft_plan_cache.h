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

#ifndef FINUFFT_PLAN_CACHE__H
#define FINUFFT_PLAN_CACHE__H

#include <array>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "finufft_plan.h"
namespace tomocam::nufft {

    template <typename T>
    class FinufftPlanCache {
      private:
        struct PlanParams {
            int dim;
            std::array<int64_t, 2> n_modes;
            int iflag;
            int ntrans;
            int device_id;

            bool operator==(const PlanParams &other) const {
                return dim == other.dim && n_modes == other.n_modes &&
                       iflag == other.iflag && ntrans == other.ntrans &&
                       device_id == other.device_id;
            }
        };

        FinufftPlanWrapper<T> type1_plan_;
        FinufftPlanWrapper<T> type2_plan_;
        std::mutex type1_mutex_;
        std::mutex type2_mutex_;
        std::optional<PlanParams> type1_params_;
        std::optional<PlanParams> type2_params_;

      public:
        FinufftPlanCache() = default;
        ~FinufftPlanCache() = default;

        FinufftPlanCache(const FinufftPlanCache &) = delete;
        FinufftPlanCache &operator=(const FinufftPlanCache &) = delete;
        FinufftPlanCache(FinufftPlanCache &&) = delete;
        FinufftPlanCache &operator=(FinufftPlanCache &&) = delete;

        // NOTE: the returned plan is only valid to use while no other call
        // with different params can race with it. Safe as long as each
        // device is only ever driven by a single host thread (true for
        // every current caller), since then a rebuild for device N can
        // never race with an in-flight use of device N's own plan.
        FinufftPlanWrapper<T> &get_plan(int type, int dim,
                                        std::array<int64_t, 2> n_modes, int iflag,
                                        int ntrans, int device_id) {
            PlanParams params{dim, n_modes, iflag, ntrans, device_id};

            if (type == 1) {
                std::lock_guard<std::mutex> lock(type1_mutex_);
                if (!type1_params_ || !(*type1_params_ == params)) {
                    type1_plan_.make_plan(1, dim, n_modes, iflag, ntrans, device_id);
                    type1_params_ = params;
                }
                return type1_plan_;
            } else if (type == 2) {
                std::lock_guard<std::mutex> lock(type2_mutex_);
                if (!type2_params_ || !(*type2_params_ == params)) {
                    type2_plan_.make_plan(2, dim, n_modes, iflag, ntrans, device_id);
                    type2_params_ = params;
                }
                return type2_plan_;
            } else {
                throw std::invalid_argument(
                    "Only FINUFFT type 1 and 2 are supported");
            }
        }
    };

    namespace plans {
        template <typename T>
        inline FinufftPlanCache<T> cache;
    }

} // namespace tomocam::nufft

#endif // FINUFFT_PLAN_CACHE__H
