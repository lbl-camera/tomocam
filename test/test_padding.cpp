#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>

#include "dist_array.h"
#include "tomocam.h"
#include "dev_array.h"
#include "internals.h"

uint64_t millisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {

    constexpr int num_slices = 4;
    constexpr int num_theta = 400;
    constexpr int num_rays = 2048;

    tomocam::dim3_t d0 = {num_slices, num_rays, num_rays};
    tomocam::DArray<float> image(d0);
	image.init(1);

    // padding
    int ipad = num_rays;
    int3 padding = {0, ipad, ipad};

    // cudaStreams
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // methods1
    auto p = image.create_partitions(1)[0];

    auto t0 = millisec();
    auto d_arr1 = tomocam::DeviceArray_fromHostR2C(p, s1);
    tomocam::addPadding(d_arr1, ipad, 2, s1);
    cudaStreamSynchronize(s1);
    auto t1 = millisec();
    // method2
    tomocam::dev_arrayf dtemp = tomocam::DeviceArray_fromHost(p, s2);
    auto d_arr2 = tomocam::add_paddingR2C(dtemp, padding, s2);
    cudaStreamSynchronize(s2);
    auto t2 = millisec();

    std::cout << "method 1 time: " << (t1-t0) << ", method 2 time: " << (t2-t1) << std::endl;    
    

    cuComplex_t * ptr1 = (cuComplex_t *) malloc(sizeof(cuComplex_t) * d_arr1.size());
    cudaMemcpyAsync(ptr1, d_arr1.dev_ptr(), sizeof(cuComplex_t) * d_arr1.size(), cudaMemcpyDeviceToHost, s1);

    cuComplex_t * ptr2 = (cuComplex_t *) malloc(sizeof(cuComplex_t) * d_arr2.size());
    cudaMemcpyAsync(ptr2, d_arr2.dev_ptr(), sizeof(cuComplex_t) * d_arr1.size(), cudaMemcpyDeviceToHost, s2);
  
    for (int i = 0; i < d_arr1.size(); i++) {
        float dx = ptr1[i].x - ptr2[i].x;
        if (dx != 0) {
            std::cout << "i = " << i << std::endl;
            std::cout << "p1 = (" << ptr1[i].x << ", " << ptr1[i].y << ")" << std::endl;
            std::cout << "p2 = (" << ptr2[i].x << ", " << ptr2[i].y << ")" << std::endl;
            std::cout << "failed ... " << std::endl;
            std::exit(1);
        }
    }
    return 0;
}
