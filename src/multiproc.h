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

#include <mpi.h>
#include <stdexcept>
#include <type_traits>

#ifndef MULTIPROC_H
#define MULTIPROC_H

#define MPI_CHECK(ans) { _mpiAssert((ans), __FILE__, __LINE__); }
inline void _mpiAssert(int code, const char *file, int line) {
    if (code != MPI_SUCCESS) {
        fprintf(stderr, "MPI error code (%d) at %s: %d\n",
                code, file, line);
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}
 
namespace tomocam {

 

    template <typename T>
    constexpr MPI_Datatype MPItype() {
        if (std::is_same<T, float>::value) {
            return MPI_FLOAT;
        } else if (std::is_same<T, double>::value) {
            return MPI_DOUBLE;
        } else if (std::is_same<T, int>::value) {
            return MPI_INT;
        } else {
            throw std::runtime_error("Unsupported data type");
        }
    }

    class MultiProc {
      private:
        int myrank_;
        int nprocs_;
        bool owned_;
        bool is_first_;
        bool is_last_;

      public:
        MultiProc() {
            int initialized = 0;
            MPI_CHECK(MPI_Initialized(&initialized));
            if (!initialized){
                int argc = 0;
                char **argv = nullptr;
                MPI_CHECK(MPI_Init(&argc, &argv));
                owned_ = true;
            } else {
                owned_ = false;
            }
    
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
            MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
            is_first_ = (myrank_ == 0);
            is_last_ = (myrank_ == nprocs_ - 1);
        }
         
        // Finalize, if owned_
        ~MultiProc() {
            if (owned_) { MPI_CHECK(MPI_Finalize()); }
        }

        // access private members
        int myrank() const { return myrank_; }
        int nprocs() const { return nprocs_; }
        bool first() const { return is_first_; }
        bool last() const { return is_last_; }

        // Wrapper for MPI_Barrier
        void Wait() { MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); }

        // Wrapper for MPI_Bcast
        template <typename T>
        void Bcast(T *buf, size_t count, int root) {
            MPI_CHECK(MPI_Bcast(buf, count, MPItype<T>(), root, MPI_COMM_WORLD));
        }

        // Wrapper for MPI_Allreduce (MPI_SUM)
        template <typename T>
        T SumReduce(T value) {
            T result;
            MPI_CHECK(MPI_Allreduce(&value, &result, 1, MPItype<T>(), MPI_SUM,
                MPI_COMM_WORLD));
            return result;
        }

        // Wrapper for MPI_Allreduce (MPI_MAX)
        template <typename T>
        T MaxReduce(T value) {
            T result;
            MPI_CHECK(MPI_Allreduce(&value, &result, 1, MPItype<T>(), MPI_MAX,
                MPI_COMM_WORLD));
            return result;
        }

        // Wrapper for MPI_Allreduce (MPI_MIN)
        template <typename T>
        T MinReduce(T value) {
            T result;
            MPI_CHECK(MPI_Allreduce(&value, &result, 1, MPItype<T>(), MPI_MIN,
                MPI_COMM_WORLD));
            return result;
        }

        // send wrapper
        template <typename T>
        void Send(const T *buf, size_t count, int proc) {
            MPI_CHECK(MPI_Send(buf, count, MPItype<T>(), proc, 123, MPI_COMM_WORLD));
        }

        // recv wrapper
        template <typename T>
        void Recv(T *buf, size_t count, int proc) {
            MPI_CHECK(MPI_Recv(buf, count, MPItype<T>(), proc, 123,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }

        // Gather wrapper
        template <typename T>
        void Gather(const T *sendbuf, size_t sendcount, T *recvbuf, size_t recvcount, int root) {
            MPI_CHECK(MPI_Gather(sendbuf, sendcount, MPItype<T>(), 
                recvbuf, recvcount, MPItype<T>(), root, MPI_COMM_WORLD));
        }

        // Gatherv wrapper
        template <typename T>
        void Gatherv(const T *sendbuf, size_t sendcount, T *recvbuf, 
                     const int *recvcounts, const int *displs, int root) {
            MPI_CHECK(MPI_Gatherv(sendbuf, sendcount, MPItype<T>(),
                recvbuf, recvcounts, displs, MPItype<T>(), root, MPI_COMM_WORLD));
        }
    };

    namespace multiproc {
        inline MultiProc mp;
    } // namespace multiproc
} // namespace tomocam
#endif // MULTIPROC_H
