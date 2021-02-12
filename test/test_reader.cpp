#include <iostream>
#include <fstream>

#include "reader.h"

const char * FILENAME = "/home/dkumar/data/phantom/phantom_00016/phantom_00016.h5";
const char * DATASET = "projs";

int main() {

    tomocam::H5Reader reader(FILENAME); 
    reader.setDataset(DATASET);
    auto sino = reader.read_sinogram(2);
    return 0;
}
