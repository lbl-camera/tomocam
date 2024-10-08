
#include <chrono>
#include <iostream>

#ifndef TIMEIT_H

class Timer {
    private:
      std::chrono::high_resolution_clock::time_point start_;
      std::chrono::milliseconds duration_;

    public:
      Timer() : duration_(0) {}

      void start() { start_ = std::chrono::high_resolution_clock::now(); }

      void stop() {
          auto end = std::chrono::high_resolution_clock::now();
          duration_ += std::chrono::duration_cast<std::chrono::milliseconds>(
              end - start_);
      }

      void reset() { duration_ = std::chrono::milliseconds(0); }
      int ms() { return duration_.count(); }

};

#endif // TIMEIT_H
