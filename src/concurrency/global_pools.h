/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef TOMOCAM_GLOBAL_POOLS__H
#define TOMOCAM_GLOBAL_POOLS__H

#include <vector>

#include "concurrency/cuda_stream.h"
#include "memory/pinned_buffer.h"
#include "concurrency/pool.h"

namespace tomocam::global {
    using tomocam::concurrency::Pool;
    
    inline std::vector<Pool<tomocam::gpu::CudaStream>> stream_pools;
    inline std::vector<Pool<tomocam::memory::PinnedBuffer>> pinned_buffer_pools;
} // namespace tomocam::global

#endif // TOMOCAM_GLOBAL_POOLS__H
