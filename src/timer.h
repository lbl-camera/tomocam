#include <chrono>
#include <stdexcept>

using std::chrono::high_resolution_clock;
typedef std::chrono::high_resolution_clock::time_point time_point_t;
typedef std::chrono::duration<double, std::milli> delta_time_t;


class Timer {
    private:
        time_point_t time_stamp_;
        delta_time_t dt_;

    public:
        Timer(): time_stamp_(high_resolution_clock::now()), dt_(0) {}
        void start() { time_stamp_ = high_resolution_clock::now(); }
        void stop() { dt_ += high_resolution_clock::now() - time_stamp_; }
        double millisec() const { return dt_.count(); }
}; 

