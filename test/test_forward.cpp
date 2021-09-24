#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "tomocam.h"

int main(int argc, char **argv) {

    constexpr int num_slices = 128;
    constexpr int num_angles = 400;
    constexpr int num_rays = 2048;

    tomocam::dim3_t d0 = {num_slices, num_rays, num_rays};
    tomocam::DArray<float> image(d0);
	image.init(1);

    tomocam::dim3_t d1 = {num_slices, num_angles, num_rays};
    tomocam::DArray<float> sino(d1);
    
    
    float * angles = new float[num_angles];
	for (int i = 0; i < num_angles; i++) 
		angles[i] = M_PI * static_cast<float>(i) / static_cast<float>(num_angles-1);

    
    float center = num_rays / 2;
    float oversample = 2;

    tomocam::radon(image, sino, angles, center, oversample);


    std::fstream fp("output.bin", std::ios::out | std::ios::binary);
    fp.write((char *) sino.data(), sino.bytes());

    delete [] angles; 
    return 0;
}
