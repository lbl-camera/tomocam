#include <iostream>
#include <fstream>


#include "dist_array.h"
#include "tomocam.h"

#include "optimize.h"

#include "reader.h"
#include "timer.h"

const int MAX_ITERS = 10;
const char * FILENAME = "/home/dkumar/data/shepp_logan/shepp_logan.h5";
const char * DATASET = "projs";
const char * ANGLES = "angs";
const int NSLICES = 1;
int main(int argc, char **argv) {

    // read data
	tomocam::H5Reader fp(FILENAME);
	fp.setDataset(DATASET);
	auto sino = fp.read_sinogram(NSLICES, 0);
	std::vector<float> angs = fp.read_angles(ANGLES);
	float * angles = angs.data();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2(d1.x, d1.z, d1.z);

    int max_iters = MAX_ITERS;
    float center = 256;
    float sigma = 10;
    float p = 1.2;

    if (argc == 2) {
        sigma = std::atof(argv[1]);
    }

    std::cout << "Input size: (" << d1.x << ", " << d1.y << ", " << d1.z <<" )" << std::endl;
	std::cout << "Center: " << center << std::endl;
	std::cout << "Smoothness: " << sigma << std::endl;
	std::cout << "No. of iterations: " << max_iters << std::endl;

    // normalize
    float tol =0.001;
    float step = 0.1;
    float penalty = 1;
    Timer t;
	auto recon = tomocam::mbir(sino, angles, center, sigma, p, max_iters, step, tol, penalty);
    t.stop();
    std::cout << "time taken(ms): " << t.millisec() << std::endl;

    return 0;
}
