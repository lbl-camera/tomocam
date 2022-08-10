#include <iostream>
#include <fstream>
#include <ctime>

#include "reader.h"
#include "dist_array.h"
#include "tomocam.h"

const char * FILENAME = "/home/dkumar/data/shepp_logan/shepp512.bin";
const int ncols = 512;
const int nproj = 400;

int main(int argc, char **argv) {

    // read data
    float * data = new float[ncols * ncols];
    std::ifstream in(FILENAME, std::ios::binary);
    in.read((char *) data, sizeof(float) * ncols * ncols);
    in.close();

    float * theta = new float[nproj];
    for (int i = 0; i < nproj; i++)
        theta[i] = M_PI * (static_cast<float>(i) / nproj - 0.5);

    tomocam::dim3_t d1 = {1, ncols, ncols};
    tomocam::dim3_t d2 = {1, nproj, ncols};
    tomocam::DArray<float> image(d1);
    image.copy(data);
    tomocam::DArray<float> sino(d2);
    
    float center = 256;
    float oversample = 2;

    tomocam::radon(image, sino, theta, center, oversample);

    std::fstream fp("output0.bin", std::ios::out | std::ios::binary);
    fp.write((char *) sino.data(), sino.bytes());
    fp.close();
    return 0;
}
