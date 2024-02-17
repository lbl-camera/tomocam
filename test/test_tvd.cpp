#include <iostream>
#include <fstream>
#include <random>

#include "tomocam.h"
#include "dist_array.h"
#include "types.h"
#include "machine.h"

namespace tomocam {

    const float weight [3][3][3] = {
            {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}},
            {{0.037, 0.0523, 0.037}, {0.0523, 0., 0.0523}, {0.037, 0.0523, 0.037}},
            {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}}
        };

    float sgnf(float v) {
        if (fabs(v) > 0) return v / fabs(v);
        else return 0;
    }

    float d_potfun(float delta, float MRF_P, float MRF_SIGMA) {
        float g = fabs(delta) / MRF_SIGMA;
        float gprime = sgnf(delta) / MRF_SIGMA;

        float temp0 = powf(g, 2-MRF_P);
        float numer = g * gprime * (2 + MRF_P * temp0);
        float denom = powf(1 + temp0, 2);
        return (numer/denom);
    }

    // calculate contraints on CPU 
    void cpuTotalVar(DArray<float> &input, DArray<float> &output, float sigma, float mrf_p) {

        // dims
        dim3_t dims = output.dims();
        int nslc = dims.x;
        int nrow = dims.y;
        int ncol = dims.z;

        #pragma omp parallel for
        for (int i = 0; i < nslc; i++) 
            for (int j = 0; j < nrow; j++) 
                for (int k = 0; k < ncol; k++) {

                    float u = input(i, j, k);
                    for (int x = 0; x < 3; x++) 
                        for (int y = 0; y < 3; y++) 
                            for (int z = 0; z < 3; z++) {
                                float d = u - input.padded(i+x-1, j+y-1, k+z-1);
                                output(i,j,k) += weight[x][y][z] * d_potfun(d, mrf_p, sigma);
                            }
                }
    }
} // namespace tomocam

int main(int argc, char **argv) {

    float p = 1.2;
    float sigma = 0.0001;
    constexpr int n = 1024;

    float * x = new float[n];
    float * y = new float[n];
    float dx = 2.0 / static_cast<float>(n);
    for (int i = 0; i < n; i++) {
        x[i] = -1.0 + i*dx;
        y[i] = tomocam::d_potfun(x[i], sigma, p);
    }
    std::ofstream fout("dpotfunc.txt");
    for (int i = 0; i < n; i++)
        fout << x[i] << ", " << y[i] << std::endl;
    fout.close();

    // data
    int nrows = 1024;
    tomocam::dim3_t dims = {4, nrows, nrows};
    tomocam::DArray<float> a(dims);
    tomocam::DArray<float> b(dims);
    tomocam::DArray<float> c(dims);

    std::default_random_engine e(0);
    std::uniform_real_distribution<float> dist(0.0,1.0);
    
    for (int i = 0; i < a.size(); i++)
        a[i] = dist(e);

    b.init(0.f);
    c.init(0.f);

    add_total_var(a, b, p, sigma, 1.0);
    cpuTotalVar(a, c, sigma, p);
    auto d = c - b;

    b.to_file("gpu.bin");
    c.to_file("cpu.bin");
    std::cout << "Max error: " << d.max() << std::endl;        
    std::cout << "L2 error: " << d.norm() / d.size() << std::endl;        

    return 0;
    // create 
}
