#include <cstdio>

#include "dist_array.h"
#include "hdf5/writer.h"
#include "tomocam.h"

int main(int argc, char **argv) {

    constexpr int N = 2047;
    constexpr int d = 32;

    auto dims = tomocam::dim3_t(1, N, N);
    tomocam::DArray<float> x(dims);

    for (int i = 0; i < N; i++) {
        int m = i / d;
        for (int j = 0; j < N; j++) {
            int n = j / d;
            if ((m + n) & 1) { x(0, i, j) = 1.f; }
        }
    }

    auto x_1 = tomocam::downsample(x, 4);
    auto x_2 = tomocam::fftdownsamp(x, 4);

    tomocam::h5::Writer h5f("dsample.h5");
    h5f.write("original", x);
    h5f.write("nneigh", x_1);
    h5f.write("fftdown", x_2);

    return 0;
}
