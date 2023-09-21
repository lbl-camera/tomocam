
#include <iostream>
#include <fstream>
#include <ctime>

#include "dist_array.h"
#include "tomocam.h"
#include "reader.h"
#include "writer.h"
#include "timer.h"


int main(int argc, char **argv) {

    // create datase
    tomocam::dim3_t d1 = {2, 8, 8};
    tomocam::DArray<float> data(d1);
    for (int i = 0; i < data.size(); i++)
        data[i] = static_cast<float>(i);

    // start the timer
    Timer t;
    auto upsamp = tomocam::upSample(data);
    t.stop(); 

    std::cout << "time taken (ms): " << t.millisec() << std::endl;
    auto d2 = upsamp.dims();

    std::cout << "new size: " << d2.x << ", " << d2.y << ", " << d2.z << std::endl; 
    for (int i = 0; i < 16; i++)
        std::cout << upsamp[i] << ", ";
    std::cout << std::endl;

    tomocam::H5Writer h5w("upscaled.h5");
    h5w.write("data", data);
    h5w.write("upsamp", upsamp);
    return 0;
}
