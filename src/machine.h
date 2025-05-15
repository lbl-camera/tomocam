/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley National
 * Laboratory (subject to receipt of any required approvals from the U.S. Dept. of
 *  Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <vector>

#include "common.h"

#ifndef TOMOCAM_MACINE__H
#define TOMOCAM_MACINE__H

// singleton
namespace tomocam {
    class MachineConfig {
      private:
        int ndevice_;
        bool unified_;
        int slcsPerStream_;

      public:
        MachineConfig() {

            cudaGetDeviceCount(&ndevice_);
            #ifdef DEBUG
            ndevice_ = 1; // for debugging
            #endif

            //check avilable GPU for managed memory access
            std::vector<int> devices;
            for (int i = 0; i < ndevice_; i++) {
                int result = -1;
                cudaDeviceGetAttribute(&result, cudaDevAttrConcurrentManagedAccess, i);
                if (result != 0) devices.push_back(i);
            }
            if (devices.empty()) unified_ = true;
            else
                unified_ = false;
            slcsPerStream_ = 4; // slices
        }

        MachineConfig(const MachineConfig &) = delete;
        MachineConfig &operator=(const MachineConfig &) = delete;

        // setters
        void num_of_gpus(int ndev)  {
            if ((ndev > 0) && (ndev <= ndevice_)) {
                ndevice_ = ndev;
            } else {
                std::cerr << "Invalid number of GPUs. Using default: " << ndevice_ << std::endl;
            }
        }

        // getters
        int num_of_gpus() const { return ndevice_; }
        int is_unified() const { return unified_; }
        int slicesPerStream() const { return slcsPerStream_; }

        /* calculate number of sub-partitions, based on free memory */
        int num_of_partitions(dim3_t dims, size_t bytes) {

            size_t total_mem = 0;
            size_t free_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            size_t max_allowed = 0.05 * free_mem;

            size_t bytes_per_slice = bytes / dims.x;
            int slcs_per_partition = max_allowed / bytes_per_slice;
            int slcs = std::min(slcsPerStream_, slcs_per_partition);

            // number of partions
            int n_partitions = dims.x / slcs;
            if (dims.x % slcs > 0) n_partitions++;
            return n_partitions;
        }
    };

    namespace Machine {
        inline MachineConfig config;
    }

} // namespace tomocam

#endif // TOMOCAM_MACINE__H
