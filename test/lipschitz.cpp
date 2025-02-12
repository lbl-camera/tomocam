#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "tomocam.h"

int main(int argc, char **argv) {


    // read data
    const int dims[] = { 4, 1536, 1280 };
    size_t size = dims[0] * dims[1] * dims[2];

    float d_Ang = 3.141592 / (float) (dims[1] - 1);
    float * angles = new float[dims[1]];
    for (int i = 0; i < dims[1]; i++)
        angles[i] = i * d_Ang;

    tomocam::dim3_t d1 = { dims[0], dims[2], dims[2] };
    tomocam::DArray<float> x(d1);
    x.init(1);
    tomocam::dim3_t d2 = { dims[0], dims[1], dims[2] };
    tomocam::DArray<float> y(d2);
    
    //float center = 639.5;
    float center = 640.f;
    float oversample = 1.5;
    if (argc == 3) {
        center = std::atof(argv[1]);
        oversample = std::atof(argv[2]);
    }

    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    tomocam::iradon(x, y, angles, center, oversample);
    tomocam::radon(y, x, angles, center, oversample);

  
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec); 
    elapsed += (finish.tv_nsec - start.tv_nsec)/ 1000000000.0;
    std::cout << "Time taken = " << elapsed << " sec" << std::endl;
    
    return 0;
}

