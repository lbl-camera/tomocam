#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "tomocam.h"

int main(int argc, char **argv) {

    constexpr int num_slices = 1;
    constexpr int num_angles = 360;
    constexpr int num_rays = 511;

    tomocam::dim3_t d0 = {num_slices, num_rays, num_rays};
    tomocam::DArray<float> image(d0);
	image.init(1);

    std::vector<float> angles(num_angles);
    for (int i = 0; i < num_angles; i++) 
		angles[i] = M_PI * static_cast<float>(i) / static_cast<float>(num_angles-1);

    int center = num_rays / 2;

    auto p = tomocam::project<float>(image, angles, center);

    // write to hdf5
    tomocam::h5::H5Writer writer("project.h5");
    writer.write("project", p);
}
