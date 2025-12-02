
#ifndef GLOBAL_SCHEMA__H
#define GLOBAL_SCHEMA__H

#include <cuda_runtime.h>
#include <format>
#include <stdexcept>

namespace tomocam::io {
    const size_t MIN_SLICES = 4;
    /**
     * @brief Data partitioning schema for distributed tomographic reconstruction.
     * - data is streamed in chucnks of (MIN_SLICES) to each gpu.
     * - Each node has ngpus_per_node gpus.
     * - data is partitioned along the first dimension (n1).
     * - Each rank gets a contiguous chunk of data.
     * - nchunks =  n1 / MIN_SLICES  (zeros padded if not divisible)
     * - nchunks_per_node = ngpus_per_node * MIN_SLICES
     * - slices_per_node = nchunks / numnodes
     *
     */
    class GlobalSchema {
      private:
        int rank_;
        dims_t global_offsets_;
        dims_t global_dims_;
        dims_t local_dims_;

      public:
        GlobalSchema(dims_t global_dims, int rank, int size)
            : rank_(rank), global_dims_(global_dims) {

            // Determine number of GPUs per node
            int ngpus_per_node = 0;
            auto err = cudaGetDeviceCount(&ngpus_per_node);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaGetDeviceCount failed");
            }

            // calulate partion strategy

            size_t nchunks =
                (global_dims_.n1 + MIN_SLICES - 1) / MIN_SLICES; // ceil division
            size_t num_gpus = ngpus_per_node_ * size;
            size_t nchunks_per_node = nchunks / size;
            size_t remainder_chunks = nchunks % size;
            if (rank_ < remainder_chunks) { nchunks_per_node++; }

            // calculate gobal offsets and local dims
            size_t start_chunk = 0;
            for (size_t r = 0; r < rank_; r++) {
                size_t chunks_for_rank = nchunks / size;
                if (r < remainder_chunks) { chunks_for_rank++; }
                start_chunk += chunks_for_rank;
            }
            global_offsets_.n1 = start_chunk * MIN_SLICES;
            global_offsets_.n2 = 0;
            global_offsets_.n3 = 0;

            local_dims_.n1 = nchunks_per_node * MIN_SLICES;
            local_dims_.n2 = global_dims_.n2;
            local_dims_.n3 = global_dims_.n3;
        }
        ~GlobalSchema();

        void initialize();
        void cleanup();
    };

} // namespace tomocam::io
#endif // GLOBAL_SCHEMA__H
