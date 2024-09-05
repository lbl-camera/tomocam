#include <iostream>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "partition.h"

void write(tomocam::h5::Writer &writer, const std::string &name,
    const tomocam::Partition<float> &p) {
    tomocam::DArray<float> array(p.dims());
    for (int i = 0; i < p.size(); i++) array[i] = p.begin()[i];
    writer.write(name.c_str(), array);
}
    int main(int argc, char **argv) {

        // read data
        const int dims[] = {22, 4, 4};
        tomocam::dim3_t d1 = {dims[0], dims[1], dims[2]};
        tomocam::DArray<float> array(d1);

        for (int i = 0; i < d1.x; i++)
            for (int j = 0; j < d1.y; j++)
                for (int k = 0; k < d1.z; k++) array(i, j, k) = i;

        tomocam::h5::Writer writer("partition.h5");
        writer.write("data", array);
        auto p1 = tomocam::create_partitions(array, 3);
        for (int i = 0; i < p1.size(); i++) {
            write(writer, "p_" + std::to_string(i + 1), p1[i]);
        }

        // create partitions with halo
        auto p2 = tomocam::create_partitions(array, 3, 1);
        for (int i = 0; i < p2.size(); i++) {
            write(writer, "p_halo" + std::to_string(i + 1), p2[i]);
        }

        // create subpartitions with halo
        int i = 1;
        for (auto p : p2) {
            auto p3 = tomocam::create_partitions(p, 3, 1);
            std::string tmp = "p_halo" + std::to_string(i);
            i++;
            for (int j = 0; j < p3.size(); j++) {
                auto name = tmp + "_sub" + std::to_string(j + 1);
                write(writer, name, p3[j]);
            }
        }

        return 0;
}

