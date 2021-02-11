#include <iostream>

#include "dist_array.h"

int main(int argc, char **argv) {


    // read data
    const int dims[] = { 20, 4,  4 };
    size_t size = dims[0] * dims[1] * dims[2];
    float * data = new float[size];

    tomocam::dim3_t d1 = { dims[0], dims[1], dims[2] };
    tomocam::DArray<float> array(d1);

    for (int i = 0; i < d1.x; i++)
        for (int j = 0; j < d1.y; j++)
            for (int k = 0; k < d1.z; k++)
                array(i, j, k) = i;    

    auto table = array.create_partitions(4, 1);
    
    int n = 0;
    for (auto p : table) {
        std::cout << "device = " << n++ << std::endl;
        auto tab2 = p.sub_partitions(2, 1);
        for (auto p2: tab2) {
            p2.print();
            std::cout << "\n" << std::endl;
        }
    }
    
    return 0;
}

