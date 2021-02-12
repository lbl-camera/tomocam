#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "tomocam.h"

const char * FILENAME = "/home/dkumar/data/shepp_logan/sino400.bin";

int main(int argc, char **argv) {

    // read data
    const int dims[] = { 1, 400, 400 };
    size_t size = dims[0] * dims[1] * dims[2];
    float * data = new float[size];

	const int num_angles = 400;
    float * angles = new float[num_angles];
	for (int i = 0; i < num_angles; i++) 
		angles[i] = M_PI * static_cast<float>(i) / static_cast<float>(num_angles-1);

    std::ifstream fp(FILENAME);
    if (! fp.is_open()) {
        std::cerr << "error! unable to open data file." << std::endl;
        exit(1);
    }
    fp.read((char *) data, sizeof(float) * size);

    tomocam::dim3_t d1 = { dims[0], dims[1], dims[2] };
    tomocam::DArray<float> sino(d1);
    sino.init(data);
    tomocam::dim3_t d2 = { dims[0], dims[2], dims[2] };
    tomocam::DArray<float> recn(d2);
    
    //float center = 639.5;
    float center = 200;
    float oversample = 1.5;

    tomocam::iradon(sino, recn, angles, center, oversample);
  
    std::fstream out("goutput.bin", std::fstream::out);
    out.write((char *) recn.data(), recn.bytes());
    out.close();
    return 0;
}
