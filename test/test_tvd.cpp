#include <iostream>
#include <fstream>

#include "dist_array.h"
#include "tomocam.h"


int main(int argc, char **argv) {

    // read data
    const int dims[] = { 16, 1536, 1280 };
    size_t size = dims[0] * dims[1] * dims[2];

    tomocam::DArray<float> a1(dims[0], dims[2], dims[2]);
    tomocam::DArray<float> a2(dims[0], dims[2], dims[2]);

    for (int i = 0; i < a1.size(); i++) {
        a1(i) = 0.f;
        a2(i) = 1.f;
    }
    
    float p = 1.2;
    float sigma = 0.1;
    add_total_var(a1, a2, p, sigma);
    return 0;
    // create 
}
