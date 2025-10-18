
#include <chrono>
#include <iostream>
#include <random>

#ifndef UTILS__H
#define UTILS__H

class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_;
        std::chrono::milliseconds duration_;

    public:
        Timer() : duration_(0) {}

        void start() {
            start_ = std::chrono::high_resolution_clock::now();
        }

        void stop() {
            auto end = std::chrono::high_resolution_clock::now();
            duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start_);
        }

        void reset() {
            duration_ = std::chrono::milliseconds(0);
        }
        int ms() {
            return duration_.count();
        }

        double seconds() {
            return static_cast<double>(duration_.count()) / 1000.0;
        }

};

class NPRandom {
    private:
        std::mt19937 gen_;

    public:
        NPRandom() {
            gen_ = std::mt19937(5489u);
        }

        template <typename T>
        T rand() {
            int a = gen_() >> 5;
            int b = gen_() >> 6;
            return static_cast<T>((a * 67108864.0 + b) / 9007199254740992.0);
        }
};

#endif // UTILS__H
