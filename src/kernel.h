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

#ifndef TOMOCAM_KERNEL__H
#define TOMOCAM_KERNEL__H

#include <cuda.h>
#include <cuda_runtime.h>
#include "dev_array.h"

namespace tomocam {

    // kernel type
    class kernel_t : public DeviceArray<float> {
        /*
         * once again, shallow copies by design
         */
      protected:
        float radius_;
        float beta_;

      public:
        // constructor
        __host__ 
        kernel_t(): radius_(0), beta_(1) {}

        // set radius
        __host__ 
        void set_radius(float r) { radius_ = r; }

        // set beta
        __host__ 
        void set_beta(float b) { beta_ = b; }

        // get radius
        __host__ __device__ 
        float radius() const { return radius_; }

        __host__ __device__
        float beta() const { return beta_; }
       
        // farthest index in negative direction
        __device__ 
        int imin(float d) { return (int) (d - radius_) + 1; }

        // farthest index in positive direction 
        __device__ 
        int imax(float d) { return (int) (d + radius_); }

        #ifdef __NVCC__
        // kaiser-bessel window
        __device__ __forceinline__
        float weight(float x) {
            float M = 2 * radius_ + 1;
            float w = 0.f;
            if (fabsf(x) < radius_) {
                float r = 2 * x / M;
                float a = cyl_bessel_i0f(beta_ * sqrtf(1.f - r * r));
                float b = cyl_bessel_i0f(beta_);
                w = a/b;
            }
            return w;
        }
 
        // calculate weight at the distane using shared memory
        __device__ __forceinline__
        float weight(float r, int k, float * shamem) {
            float tt = (r - k);
            if (tt < 0.f)  tt = -tt;
            float x = tt * (size_ - 1)/radius_; 
            int i = (int) x;
            float dx = x - i;
            if ((i+1) < size_) return (shamem[i] * (1.f - dx) + shamem[i+1] * dx);
            else 
                return 0;
        }
 
        // calculate weight at the distane using global memory
        __device__ __forceinline__
        float weight(float r, int k) {
            float tt = (r - k);
            if (tt < 0.f)  tt = -tt;
            float x = tt * (size_ - 1)/radius_; 
            int i = (int) x;
            float dx = x - i;
            if ((i+1) < size_) return (ptr_[i] * (1.f - dx) + ptr_[i+1] * dx);
            else 
                return 0;
        }
        #endif // __NVCC__
 
        // calculate weight at the distane using shared memory
    };
} // namespace tomocam

#endif // TOMOCAM_KERNEL__H
