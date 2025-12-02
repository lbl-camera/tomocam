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


#ifndef PLANS_CACHE__H
#define PLANS_CACHE__H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cufft.h>

#include "transforms/fft.h"

namespace tomocam::transforms {
    class PlanCache {
      private:
        int gpu_id_;
        std::unodered_map<std::sting, cufftHandle_t> cufft_plans_;
        void *workspace_;
        size_t workspace_size_;

      public:
        PlanCache() : workspace_(nullptr), workspace_size_(0) {}

        ~PlanCache() {
            for (auto [key, plan] : cufft_plans_) {
                auto err = cufftDestroy(plan);
                if (err != CUFFT_SUCCESS) {
                    std::cerr << "Failed to destroy cuFFT plan: " << err
                              << std::endl;
                }
            }
        }

        /// If plan exists in cache, return it. Otherwise, create a new plan,
        /// store it in the cache, and return it.
        template <FFT_Type>
        cufftHandle_t get<FFT_Type>(size_t n1, size_t n2, size_t n3) {
            std::string key = generate_key<FFT_Type>(n1, n2, n3);
            auto it = cufft_plans_.find(key);
            if (it != cufft_plans_.end()) {
                // attach workspace
                cufftSetWorkArea(it->second, workspace_);
                return it->second;
            }
            cufftHandle_t plan = create_plan<FFT_Type>(n1, n2, n3);
            cufft_plans_[key] = plan;
            // get workspace size
            size_t ws_size;
            cufftGetSize(plan, &ws_size);
            if (ws_size > workspace_size_) {
                if (workspace_) {
                    cudaFree(workspace_);
                }
                cudaMalloc(&workspace_, ws_size);
                workspace_size_ = ws_size;
            }
            // attach workspace
            cufftSetWorkArea(plan, workspace_);
            return plan;
        }
    };
    namespace global {
        inline tomocam::transforms::PlanCache<float> *get_float_plan_cache
    }
} // namespace tomocam::transforms
#endif // PLANS_CACHE__H
