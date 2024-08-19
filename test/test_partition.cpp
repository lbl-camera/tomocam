#include <iostream>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "partition.h"

void write(tomocam::h5::H5Writer &writer, const std::string &name,
    const tomocam::Partition<float> &p) {
    tomocam::DArray<float> array(p.dims());
    for (int i = 0; i < p.size(); i++) array[i] = p.begin()[i];
    writer.write(name.c_str(), array);
}
    int main(int argc, char **argv) {

        // read data
        const int dims[] = {20, 16, 16};
        tomocam::dim3_t d1 = {dims[0], dims[1], dims[2]};
        tomocam::DArray<float> array(d1);

        for (int i = 0; i < d1.x; i++)
            for (int j = 0; j < d1.y; j++)
                for (int k = 0; k < d1.z; k++) array(i, j, k) = i;

        tomocam::h5::H5Writer writer("partition.h5");
        writer.write("data", array);
        auto p1 = tomocam::create_partitions(array, 2);
        for (int i = 0; i < p1.size(); i++) {
            write(writer, "p_" + std::to_string(i + 1), p1[i]);
        }

        // create partitions with halo
        auto p2 = tomocam::create_partitions(array, 4, 1);

        // create subpartitions with halo
        const int n2 = 3;
        write(writer, "p_halo1", p2[0]);
        auto p3 = tomocam::create_partitions(p2[0], n2, 1);
        for (int j = 0; j < p3.size(); j++) {
            write(writer, "p_halo1_sub" + std::to_string(j + 1), p3[j]);
        }

        write(writer, "p_halo2", p2[2]);
        auto p4 = tomocam::create_partitions(p2[2], n2, 1);
        for (int j = 0; j < p4.size(); j++) {
            write(writer, "p_halo2_sub" + std::to_string(j + 1), p4[j]);
        }

        auto p5 = tomocam::create_partitions(p2[3], n2, 1);

        return 0;
}

