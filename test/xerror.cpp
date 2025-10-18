
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "dist_array.h"
#include "dev_array.h"
#include "toeplitz.h"
#include "machine.h"
#include "tomocam.h"

#include "timer.h"

int main(int argc, char **argv) {

    const int nslices = 16;
    const int npixel = 2047;

    auto rng = NPRandom();

    // create data
    tomocam::DArray<float> xcurr(tomocam::dim3_t{nslices, npixel, npixel});
    for (int i = 0; i < xcurr.size(); i++) {
        xcurr[i] = rng.rand<float>();
    }
    auto xcurr_norm = xcurr.norm();
    std::cout << "|| x ||_2 " << xcurr_norm << std::endl;


    /// measure time to transfor array to device
    Timer time_to_device;
    auto p = tomocam::create_partitions(xcurr, 4);
    time_to_device.start();
    auto xcurr_dev = tomocam::DeviceArray<float>(p[0]);
    time_to_device.stop();
    std::cout << "Time to device: " << time_to_device.ms() << " ms" << std::endl;

    tomocam::Machine::config.barrier();
    // allocate solution array
    tomocam::DArray<float> xprev(xcurr.dims());
    xprev.init(1.f);

    // error 1
    Timer time1;
    time1.start();
    auto err1 = (xcurr - xprev).norm();
    time1.stop();

    // error 2
    Timer time2;
    time2.start();
    auto err2 = tomocam::xerror(xcurr, xprev);
    time2.stop();

    // compute error
    // compare
    std::cout << "Error 1: " << err1 << std::endl;
    std::cout << "Error 2: " << err2 << std::endl;
    std::cout << "Error2/Error1: " << err2 / err1 << std::endl;
    std::cout << "Time 1: " << time1.ms() << " ms" << std::endl;
    std::cout << "Time 2: " << time2.ms() << " ms" << std::endl;
    return 0;
}
