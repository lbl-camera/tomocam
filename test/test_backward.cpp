#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "tomocam.h"
#include "reader.h"
#include "timer.h"

const char * FILENAME = "/home/dkumar/data/phantom_00017/phantom_00017.h5";
const char * DATASET = "projs";
const char * ANGLES = "angs";

int main(int argc, char **argv) {

    // read data
    tomocam::H5Reader fp(FILENAME);
    fp.setDataset(DATASET);
    auto sino = fp.read_sinogram(16);
    auto angs = fp.read_angles(ANGLES);
    float * theta = angs.data();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2 = {d1.x, d1.z, d1.z};
    tomocam::DArray<float> recn(d2);
    
    float center = 640;
    float oversample = 2;

    Timer t;
    tomocam::iradon(sino, recn, theta, center, oversample);
    t.stop();
    std::cout << "time taken(ms): " << t.millisec() << std::endl;
 
    std::fstream out("backward.bin", std::fstream::out);
    out.write((char *) recn.data(), recn.bytes());
    out.close();
    return 0;
}
