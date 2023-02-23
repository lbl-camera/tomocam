#include <iostream>
#include <fstream>


#include "dist_array.h"
#include "tomocam.h"

#include "optimize.h"

#include "reader.h"
#include "timer.h"

const float PI = M_PI;
const char * FILENAME = "/home/dkumar/data/foam/foam_tomo.h5";
const char * DATASET = "projs";
const char * ANGLES = "projgeom_angles";
const int NSLICES = 4;
int main(int argc, char **argv) {

    // read data
	tomocam::H5Reader fp(FILENAME);
	fp.setDataset(DATASET);
	auto sino = fp.read_sinogram(NSLICES, 0);
	std::vector<float> angs = fp.read_angles(ANGLES);
	float * angles = angs.data();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2(d1.x, d1.z, d1.z);

    int max_iters = 50;
    float center = 1024;
    float oversample = 2;
    float sigma = 0.01;
    float p = 1.2;

    if (argc == 2) {
        sigma = std::atof(argv[1]);
    }
    float lam = 1 / p;

    std::cout << "Input size: (" << d1.x << ", " << d1.y << ", " << d1.z <<" )" << std::endl;
	std::cout << "Center: " << center << std::endl;
	std::cout << "Oversampling: " << oversample << std::endl;
	std::cout << "Smoothness: " << sigma << std::endl;
	std::cout << "No. of iterations: " << max_iters << std::endl;

    // normalize
    sino = sino * (1./ sino.max());
    Timer t;
	auto recon = tomocam::mbir(sino, angles, center, oversample, sigma, p, lam, max_iters);
    t.stop();
    std::cout << "time taken(ms): " << t.millisec() << std::endl;

    std::fstream out("recon.bin", std::ios::out | std::ios::binary);
    out.write((char *) recon.data(), recon.size() * sizeof(float));

    return 0;
}
