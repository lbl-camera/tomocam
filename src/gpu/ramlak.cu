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


#include "memory/dev_array.h"
#include "utils/types.h"

#include "gpu/dev_memory.cuh"
#include "gpu/utils.cuh"


namespace tomocam {
    namespace gpu {


        template <typename T>
        __global__ void ramlak_kernel(DeviceMemory<gpu::complex_t<T>> singal, T cutoff) {

            // indices
            dim3_t dims = singal.dims();
            T cen = static_cast<T>(dims.z / 2);
            T span = static_cast<T>(dims.z);

            int3 idx = Index3D();
            if (idx < dims) {
                T freq = abs(idx.z - cen) / span;
                T filter = 0;
                if (freq < cutoff) { filter = freq; }
                
                // apply filter
                singal[idx] = singal[idx] * filter;
            }
        }


        template <typename T>
        void apply_filter(DeviceArray<gpu::complex_t<T>> & singal) {

           // cuda compute grid
           Grid grid(singal.dims());

           // original frequency is [-0.5, 0.5]
           T cutoff = static_cast<T>(1.0);

            // apply filter 
            ramlak_kernel<T> <<<grid.blocks(), grid.threads()>>>(singal, cutoff);

            // check for errors 
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Error in ramlak_kernel: " << cudaGetErrorString(err) << std::endl;
            }
        }

        // explicit instantiation
        template void apply_filter(DeviceArray<gpu::complex_t<float>> &);
        template void apply_filter(DeviceArray<gpu::complex_t<double>> &);

    } // namespace gpu
} // namespace tomocam
