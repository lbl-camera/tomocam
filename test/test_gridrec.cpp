#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "tomocam.h"

int main(int argc, char **argv) {


    // read data
    const int dims[] = { 16, 1536, 1280 };
    size_t size = dims[0] * dims[1] * dims[2];
    float * data = new float[size];
    float * angles = new float[dims[1]];

    std::fstream ang("angles.bin", std::ifstream::in);
    if (! ang.is_open()) {
        std::cerr << "error! unable to open data file." << std::endl;
        exit(1);
    }
    ang.read((char *) angles, sizeof(float) * dims[1]);

    std::fstream fp("sino.bin", std::ifstream::in);
    if (! fp.is_open()) {
        std::cerr << "error! unable to open data file." << std::endl;
        exit(1);
    }
    fp.read((char *) data, sizeof(float) * size);

    tomocam::dim3_t d1 = { dims[0], dims[1], dims[2] };
    tomocam::DArray<float> sino(d1);
    sino.copy(data);
    tomocam::dim3_t d2 = { dims[0], dims[2], dims[2] };
    tomocam::DArray<float> recn(d2);
    
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

    tomocam::iradon(sino, recn, angles, center, oversample);
  
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec); 
    elapsed += (finish.tv_nsec - start.tv_nsec)/ 1000000000.0;
    std::cout << "Time taken = " << elapsed << " sec" << std::endl;
    std::fstream out("output.bin", std::fstream::out);
    out.write((char *) recn.data(), recn.bytes());
    out.close();
    return 0;
}
