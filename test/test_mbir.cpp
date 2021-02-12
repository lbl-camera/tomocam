#include <iostream>
#include <fstream>


#include "dist_array.h"
#include "tomocam.h"

#include "optimize.h"

#include "reader.h"

const float PI = M_PI;
const int MAX_ITERS = 150;
//const char * FILENAME = "/home/dkumar/data/dula/llzo_stack.h5";
const char * FILENAME = "/home/dkumar/data/phantom/phantom_00016/phantom_00016.h5";
const char * DATASET = "projs";
//const char * ANGLES = "angles";
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

    // normalize
    float maxval = sino.max();
    float minval = sino.min();


    #pragma omp parallel for
    for (int i = 0; i < sino.size(); i++)
        sino[i] = (sino[i] - minval)/(maxval - minval);

    tomocam::DArray<float> model(d2);
    model.init(1.f);
    tomocam::DArray<float> grad(d2);
    grad.init(0.f);


    float center = 640;
    //float center = 1279.375;
    float oversample = 1.5;
    float sigma = 1;

    if (argc == 2) {
        sigma = std::atof(argv[1]);
    }

    std::cout << "Inpit size: (" << d1.x << ", " << d1.y << ", " << d1.z <<" )" << std::endl;
	std::cout << "Center: " << center << std::endl;
	std::cout << "Oversampling: " << oversample << std::endl;
	std::cout << "Smoothness: " << sigma << std::endl;

    tomocam::Optimizer opt(d2, d1, angles, center, oversample, sigma);
    for (int i = 0; i < MAX_ITERS; i++) {
        grad = model;
        tomocam::gradient(grad, sino, angles, center, oversample);
        std::cout << "Error = " << grad.norm() << std::endl;
	    tomocam::add_total_var(model, grad, 1.2, sigma);
        opt.update(model, grad);
    }

    std::fstream out("output.bin", std::fstream::out);
    out.write((char *) model.data(), model.size() * sizeof(float));

    return 0;
}
