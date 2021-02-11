#include <iostream>
#include <fstream>


#include "dist_array.h"
#include "tomocam.h"

#include "optimize.h"

#include "reader.h"

const float PI = M_PI;
const char * FILENAME = "/home/dkumar/data/shepp_logan/sino400.bin";
const char * DATASET = "projs";

const int dims[] = { 1, 400, 400 };
const int num_angles = 400;

int main(int argc, char **argv) {

    // read data
    size_t size = dims[0] * dims[1] * dims[2];
    float * data = new float[size];
    float * angles = new float[num_angles];


	float d_angle = M_PI / (float) (num_angles-1);
    for (int i = 0; i < dims[1]; i++) 
		angles[i] = i * d_angle;

    std::ifstream fp(FILENAME);
    if (! fp.is_open()) {
        std::cerr << "error! unable to open data file." << std::endl;
        exit(1);
    }
    fp.read((char *) data, sizeof(float) * size);

    tomocam::dim3_t d1(dims[0], num_angles, dims[2]);
    tomocam::dim3_t d2(dims[0], dims[1], dims[2]);

    tomocam::DArray<float> sino(d1);
    sino.init(data);

    // normalize
    float maxval = sino.max();
    float minval = sino.min();

    std::cout << "Size = " << sino.size() << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < sino.size(); i++)
        sino[i] = (sino[i] - minval)/(maxval - minval);

    tomocam::DArray<float> model(d2);
    model.init(1.f);
    tomocam::DArray<float> grad(d2);
    grad.init(0.f);


    float center = 200;
    float oversample = 1.5;
    float sigma = 10;

    if (argc == 3) {
        center = std::atof(argv[1]);
        oversample = std::atof(argv[2]);
    }

    tomocam::Optimizer opt(d2, d1, angles, center, oversample, sigma);
    for (int i = 0; i < 60; i++) {
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
