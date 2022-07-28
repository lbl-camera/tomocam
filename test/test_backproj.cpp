#include <iostream>
#include <fstream>
#include <ctime>

#include "reader.h"
#include "dist_array.h"
#include "tomocam.h"

const char * FILENAME = "/home/dkumar/data/phantom_00017/phantom_00017.h5";
const char * DATASET = "projs";
const char * ANGLES = "angs";

int main(int argc, char **argv) {

    // read data
    tomocam::H5Reader h5fp(FILENAME);
    h5fp.setDataset(DATASET);
    auto sino = h5fp.read_sinogram(16, 500);
    auto angs = h5fp.read_angles(ANGLES);
    float * theta = angs.data();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2 = {d1.x, d1.z, d1.z};
    tomocam::DArray<float> recn(d2);
    
    float center = 640;
    float oversample = 2;

    tomocam::iradon(sino, recn, theta, center, oversample);

    std::fstream fp("output0.bin", std::ios::out | std::ios::binary);
    fp.write((char *) sino.data(), sino.bytes());
    fp.close();
    return 0;
}
