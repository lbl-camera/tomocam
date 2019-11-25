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

#ifndef TOMOCAM_MACINE__H
#define TOMOCAM_MACINE__H

// singleton
namespace tomocam {
    class MachineConfig {
      private:
        int  ndevice_;
        bool unified_;
        int nStreams_;
        unsigned slcsPerStream_;
        MachineConfig() {
            cudaGetDeviceCount(&ndevice_);
            //    check avilable GPU for managed memory access
            std::vector<unsigned> devices;
            for (unsigned i = 0; i < ndevice_; i++) {
                int result = -1;
                cudaDeviceGetAttribute(&result,
                                       cudaDevAttrConcurrentManagedAccess, i);
                if (result != 0) device_id_.push_back(i);
            }
            if (device_id_.empty())
                unified_ = true;
            else
                unified_ = false;
            nStreams_ = 3;
            workPerStream_ = 10; // slices 
        }

      public:
        MachineConfig(const MachineConfig &) = delete;
        MachineConfig & operator=(const MachineConfig &) = delete;
        static MachineConfig &getInstance() {
            static MachineConfig instance;
            return instance;
        }
        int num_of_gpus() const { return ndevice_; }
        int is_unified() const { return unified_; }
    };
#endif // TOMOCAM_MACINE__H
} // namespace
