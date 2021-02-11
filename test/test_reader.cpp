#include <iostream>
#include <fstream>

#include "reader.h"
constexpr size_t SIZE = 2 * 1536 * 1280;
constexpr size_t BYTES = SIZE * sizeof(float);

const char * FILENAME = "/home/dkumar/Data/phantom/phantom_00016/phantom_00016.h5";
const char * DATASET = "projs";


int main() {

    tomocam::H5Reader reader(FILENAME); 
    reader.setDataset(DATASET);
    auto sino = reader.read_sinogram(2);
    std::ofstream out("sino.bin", std::ios::out | std::ios::binary);
    out.write((char *) sino.data(), BYTES);
    return 0;
}
