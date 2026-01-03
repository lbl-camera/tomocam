#include <ctime>
#include <fstream>
#include <iostream>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "timer.h"
#include "tomocam.h"

int main(int argc, char **argv) {

    constexpr int num_slices = 16;
    constexpr int num_angles = 220;
    constexpr int num_rays = 2047;

    tomocam::dim3_t d0 = {num_slices, num_rays, num_rays};
    tomocam::DArray<float> image(d0);
    auto rng = NPRandom();

    for (int i = 0; i < image.size(); i++) image[i] = rng.rand<float>();

    std::vector<float> angles(num_angles);
    for (int i = 0; i < num_angles; i++)
        angles[i] =
            M_PI * static_cast<float>(i) / static_cast<float>(num_angles - 1);

    auto p = tomocam::project<float>(image, angles);

    // write to hdf5
    tomocam::h5::Writer writer("project.h5");
    writer.write("project", p);
}
