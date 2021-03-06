#include <iostream>
#include <fstream>


#include "dist_array.h"
#include "tomocam.h"

#include "optimize.h"

#include "reader.h"

const float PI = M_PI;
const int MAX_ITERS = 150;
const char * FILENAME = "/home/dkumar/data/phantom/phantom_00016/phantom_00016.h5";
const char * DATASET = "projs";
const char * ANGLES = "angs";

int main(int argc, char **argv) {

    // read data
	tomocam::H5Reader fp(FILENAME);
	fp.setDataset(DATASET);
	auto sino = fp.read_sinogram(16);
	std::vector<float> angs = fp.read_angles(ANGLES);
	float * angles = angs.data();

    tomocam::dim3_t d1 = sino.dims();
    tomocam::dim3_t d2(d1.x, d1.z, d1.z);
	tomocam::DArray<float> model(d2);

    float center = 640;
    float oversample = 1.5;
    float sigma = 1;
    float p = 1.2;

    if (argc == 2) {
        sigma = std::atof(argv[1]);
    }

    std::cout << "Inpit size: (" << d1.x << ", " << d1.y << ", " << d1.z <<" )" << std::endl;
	std::cout << "Center: " << center << std::endl;
	std::cout << "Oversampling: " << oversample << std::endl;
	std::cout << "Smoothness: " << sigma << std::endl;

	tomocam::mbir(sino, model, angles, center, 100, oversample, sigma, p);

    std::fstream out("output.bin", std::fstream::out);
    out.write((char *) model.data(), model.size() * sizeof(float));

    return 0;
}
