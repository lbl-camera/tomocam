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

namespace tomocam {

    template <typename T>
    constexpr MPI_Datatype MPItype() {
        if (std::is_same<T, float>::value) {
            return MPI_FLOAT;
        }  else if (std::is_same<T, double>::value) {
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
            bool is_first_;
            bool is_last_;

        public:
            MultiProc(): myrank_(0), nprocs_(1), is_first_(true), is_last_(true) {}

            // initialize MPI
            void init(int argc, char **argv) {
                MPI_Init(&argc, &argv);
                MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
                MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
                is_first_ = (myrank_ == 0);
                is_last_ = (myrank_ == nprocs_ - 1);
            }

            // Finalize
            void finalize() {
                MPI_Finalize();
            }

            // access private members
            int myrank() const {
                return myrank_;
            }
            int nprocs() const {
                return nprocs_;
            }
            bool first() const {
                return is_first_;
            }
            bool last() const {
                return is_last_;
            }

            // Wrapper for MPI_Barrier
            void Wait() {
                MPI_Barrier(MPI_COMM_WORLD);
            }

            // Wrapper for MPI_Bcast
            template <typename T>
            void Bcast(T *buf, size_t count, int root) {
                MPI_Bcast(buf, count, MPItype<T>(), root, MPI_COMM_WORLD);
            }

            // Wrapper for MPI_Allreduce (MPI_SUM)
            template <typename T>
            T SumReduce(T value) {
                T result;
                MPI_Allreduce(&value, &result, 1, MPItype<T>(), MPI_SUM,
                    MPI_COMM_WORLD);
                return result;
            }

            // Wrapper for MPI_Allreduce (MPI_MAX)
            template <typename T>
            T MaxReduce(T value) {
                T result;
                MPI_Allreduce(&value, &result, 1, MPItype<T>(), MPI_MAX,
                    MPI_COMM_WORLD);
                return result;
            }

            // send wrapper
            template <typename T>
            void Send(const T *buf, size_t count, int proc) {
                auto st =
                    MPI_Send(buf, count, MPItype<T>(), proc, 123, MPI_COMM_WORLD);
                if (st != MPI_SUCCESS) {
                    throw std::runtime_error("MPI_Send failed");
                }
            }

            // recv wrapper
            template <typename T>
            void Recv(T *buf, size_t count, int proc) {
                auto st = MPI_Recv(buf, count, MPItype<T>(), proc, 123,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (st != MPI_SUCCESS) {
                    throw std::runtime_error("MPI_Recv failed");
                }
            }
    };

    namespace multiproc {
        inline MultiProc mp;
    } // namespace multiproc
} // namespace tomocam
#endif // MULTIPROC_H
