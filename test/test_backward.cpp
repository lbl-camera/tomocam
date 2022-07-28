
#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "dev_array.h"
#include "tomocam.h"
#include "reader.h"
#include "timer.h"
#include "fft.h"
#include "internals.h"
#include "nufft.h"

const char * FILENAME = "/home/dkumar/data/phantom_00017/phantom_00017.h5";
const char * DATASET = "projs";
const char * ANGLES = "angs";

int main(int argc, char **argv) {

    // read data
    tomocam::H5Reader fp(FILENAME);
    fp.setDataset(DATASET);
    auto sino = fp.read_sinogram(16, 500);
    auto angs = fp.read_angles(ANGLES);
    float * theta = angs.data();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2 = {d1.x, d1.z, d1.z};
    tomocam::DArray<float> recn(d2);
    
    float center = 640;
    float oversample = 2;

    // padding for oversampling
    int ipad = (int) ((oversample-1) * d1.z / 2);
    int3 pad1 = {0, 0, ipad};
    int3 pad2 = {0, ipad, ipad};
    center += ipad;
    tomocam::dim3_t padded_dims(d2.x, d2.y + 2 * ipad, d2.z + 2 * ipad);

    // nufft grid
    int ncols = d1.z + 2 * ipad;
    int nproj = d1.y;
    tomocam::NUFFTGrid grid(ncols, nproj, theta, center, 0);

    // create partitions
    auto part1 = sino.create_partitions(1);
    auto part2 = recn.create_partitions(1);

    // move data to GPU RAM
    auto temp = tomocam::DeviceArray_fromHost(part1[0], 0);
    auto d_sino = add_paddingR2C(temp, pad1, 0);
    auto d_recn = tomocam::DeviceArray_fromDims<cuComplex_t>(padded_dims, 0);
    
    // start the timer
    Timer t;
    tomocam::back_project(d_sino, d_recn, center, grid);
    auto temp2 = tomocam::remove_paddingC2R(d_recn, pad2, 0);
    t.stop(); 

    std::cout << "time taken (ms): " << t.millisec() << std::endl;
 
    tomocam::copy_fromDeviceArray(part2[0], temp2, 0);
    std::fstream out("backward0.bin", std::fstream::out);
    out.write((char *) recn.data(), recn.bytes());
    out.close();
    return 0;
}
