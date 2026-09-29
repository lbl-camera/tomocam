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

#include <chrono>
#include <iostream>

#ifndef TIMERS__H
#define TIMERS__H

namespace tomocam {
    class Timer {
      private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
        std::chrono::duration<double> elapsed_time_;

      public:
        Timer() {
            start_time_ = std::chrono::high_resolution_clock::now();
            elapsed_time_ = std::chrono::duration<double>(0);
        }

        void start() {
            elapsed_time_ += std::chrono::high_resolution_clock::now() - start_time_;
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        void stop() {
            auto tnow = std::chrono::high_resolution_clock::now();
            elapsed_time_ += tnow - start_time_;
        }

        void pause() {
            auto tnow = std::chrono::high_resolution_clock::now();
            elapsed_time_ += tnow - start_time_;
        }

        double ms() const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                       elapsed_time_)
                .count();
        }

        double seconds() const {
            return std::chrono::duration_cast<std::chrono::seconds>(elapsed_time_)
                .count();
        }
    };

} // namespace tomocam
#endif //
