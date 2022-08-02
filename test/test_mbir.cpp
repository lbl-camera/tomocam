#include <iostream>
#include <fstream>


#include "dist_array.h"
#include "tomocam.h"

#include "optimize.h"

#include "reader.h"
#include "timer.h"

const float PI = M_PI;
const int MAX_ITERS = 150;
const char * FILENAME = "/home/dkumar/data/phantom_00017/phantom_00017.h5";
const char * DATASET = "projs";
const char * ANGLES = "angs";
const int NSLICES = 16;
int main(int argc, char **argv) {

    // read data
	tomocam::H5Reader fp(FILENAME);
	fp.setDataset(DATASET);
	auto sino = fp.read_sinogram(NSLICES);
	std::vector<float> angs = fp.read_angles(ANGLES);
	float * angles = angs.data();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2(d1.x, d1.z, d1.z);

    int max_iters = 10;
    float center = 640;
    float oversample = 2;
    float sigma = 0.1;
    float p = 1.2;

    if (argc == 2) {
        sigma = std::atof(argv[1]);
    }

    std::cout << "Input size: (" << d1.x << ", " << d1.y << ", " << d1.z <<" )" << std::endl;
	std::cout << "Center: " << center << std::endl;
	std::cout << "Oversampling: " << oversample << std::endl;
	std::cout << "Smoothness: " << sigma << std::endl;
	std::cout << "No. of iterations: " << max_iters << std::endl;

    Timer t;
	auto recon = tomocam::mbir(sino, angles, center, oversample, sigma, p, max_iters);
    t.stop();
    std::cout << "time taken(ms): " << t.millisec() << std::endl;

    char foutname[16];
    sprintf(foutname, "recon%6.4f.bin", sigma); 
    std::fstream out(foutname, std::fstream::out);
    out.write((char *) recon.data(), recon.size() * sizeof(float));

    return 0;
}
