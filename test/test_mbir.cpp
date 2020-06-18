#include <iostream>
#include <fstream>

#include "dist_array.h"
#include "tomocam.h"

const float PI = 3.141592f;

int main(int argc, char **argv) {

    // read data
    const int dims[] = { 16, 1536, 1280 };
    size_t size = dims[0] * dims[1] * dims[2];
    float * data = new float[size];
    float * angles = new float[dims[1]];


	float d_angle = PI / (float) (dims[1]);
    for (int i = 0; i < dims[1]; i++) 
		angles[i] = i * d_angle;

    std::fstream fp("sino_00016.bin", std::ifstream::in);
    if (! fp.is_open()) {
        std::cerr << "error! unable to open data file." << std::endl;
        exit(1);
    }
    fp.read((char *) data, sizeof(float) * size);

    tomocam::dim3_t d1(dims[0], dims[1], dims[2]);
    tomocam::dim3_t d2(dims[0], dims[2], dims[2]);

    tomocam::DArray<float> sino(d1);
    sino.copy(data);
    tomocam::DArray<float> model(d2);
    tomocam::DArray<float> grad(d2);

    #pragma omp parallel for
    for (int i = 0; i < model.size(); i++) model[i] = 1.f;
    #pragma omp parallel for
    for (int i = 0; i < grad.size(); i++) grad[i] = 0.f;

    
    
    float center = 640;
    float oversample = 1.5;
    if (argc == 3) {
        center = std::atof(argv[1]);
        oversample = std::atof(argv[2]);
    }

    for (int i = 0; i < 50; i++) {
        grad = model;
        tomocam::gradient(grad, sino, angles, center, oversample);
        tomocam::add_total_var(model, grad, 1.2, 0.1);
        tomocam::axpy(-0.2, grad, model);  
        float err = tomocam::norm2(grad);
        std::cout << "eror = " << err << std::endl; 
    }

    
    std::fstream out("output.bin", std::fstream::out);
    out.write((char *) model.data(), model.size() * sizeof(float));

    return 0;
}
