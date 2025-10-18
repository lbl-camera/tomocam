#include <iostream>
#include <fstream>
#include <random>

#include "dist_array.h"
#include "gpu/totalvar.cuh"
#include "machine.h"
#include "tomocam.h"
#include "types.h"
#include "timer.h"

namespace tomocam {

    const float weight[3][3][3] = {
        {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}},
        {{0.037, 0.0523, 0.037}, {0.0523, 0., 0.0523}, {0.037, 0.0523, 0.037}},
        {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}}
    };

    const float MRF_Q = 2.f;
    const float MRF_C = 0.001;

    float d_potfun(float delta, float sigma, float p) {
        float sigma_q   = std::pow(sigma, MRF_Q);
        float sigma_q_p = std::pow(sigma, MRF_Q - p);

        float temp1 = std::pow(std::abs(delta), MRF_Q - p) / sigma_q_p;
        float temp2 = std::pow(std::abs(delta), MRF_Q - 1);
        float temp3 = MRF_C + temp1;

        if (delta > 0.f)
            return ((temp2 / (temp3 * sigma_q)) * (MRF_Q - ((MRF_Q - p) * temp1) / temp3));
        else if (delta < 0.f)
            return ((-1 * temp2 / (temp3 * sigma_q)) * (MRF_Q - ((MRF_Q - p) * temp1) / temp3));
        else
            return 0;
    }

    // calculate contraints on CPU
    void cpuTotalVar(DArray<float> &input, DArray<float> &output, float sigma, float mrf_p) {

        // dims
        dim3_t dims = output.dims();
        int nslc = dims.x;
        int nrow = dims.y;
        int ncol = dims.z;

        #pragma omp parallel for
        for (int i = 0; i < nslc; i++) {
            for (int j = 0; j < nrow; j++) {
                for (int k = 0; k < ncol; k++) {
                    float u = input(i, j, k);
                    float v = 0.f;
                    for (int z = 0; z < 3; z++) {
                        for (int y = 0; y < 3; y++) {
                            for (int x = 0; x < 3; x++) {
                                int i1 = i + z - 1;
                                if (i1 < 0) i1 = 0;
                                if (i1 >= nslc) i1 = nslc - 1;
                                int j1 = j + y - 1;
                                if (j1 < 0) j1 = 0;
                                if (j1 >= nrow) j1 = nrow - 1;
                                int k1 = k + x - 1;
                                if (k1 < 0) k1 = 0;
                                if (k1 >= ncol) k1 = ncol - 1;
                                float d = u - input(i1, j1, k1);
                                v += weight[z][y][x] * d_potfun(d, sigma, mrf_p);
                            }
                        }
                    }
                    output(i, j, k) = v;
                }
            }
        }
    }
} // namespace tomocam

int main(int argc, char **argv) {

    float p = 1.2;
    float sigma = 10.0;

    // Random number generator
    auto rng = NPRandom();

    // data
    std::cout << "allocating memory .. " << std::endl;
    tomocam::dim3_t dims = {16, 1023, 1023};
    tomocam::DArray<float> a(dims);
    tomocam::DArray<float> b(dims);
    tomocam::DArray<float> c(dims);
    tomocam::DArray<float> d(dims);

    // initializing
    for (int i = 0; i < a.size(); i++) {
        a[i] = rng.rand<float>();
    }
    std::cout << "initializing ... " << std::endl;
    b.init(0.f);
    c.init(0.f);
    d.init(0.f);

    // test GPU code
    /*
    std::cout << "testing single GPU code ... " << std::endl;
    tomocam::DeviceArray<float> d_a(dims);
    cudaMemcpy(d_a.dev_ptr(), a.begin(), a.bytes(), cudaMemcpyHostToDevice);
    tomocam::DeviceArray<float> d_b(dims);
    cudaMemset(d_b.dev_ptr(), 0, d_b.bytes());

    tomocam::gpu::add_total_var2(d_a, d_b, sigma, p);
    // copy back to host
    cudaMemcpy(b.begin(), d_b.dev_ptr(), b.bytes(), cudaMemcpyDeviceToHost);
    */

    std::cout << "testing CPU code ... " << std::endl;

    Timer t0; 
    t0.start();
    cpuTotalVar(a, b, sigma, p);
    t0.stop();

    
    // test multi-GPU code
    std::cout << "\n\ntesting multi-GPU code ... " << std::endl;
    Timer t1;
    t1.start();
    tomocam::add_total_var(a, c, sigma, p);
    t1.stop();
    Timer t2;
    t2.start();
    tomocam::add_total_var2(a, d, sigma, p);
    t2.stop();

    // report times
    std::cout << "CPU time: " << t0.ms() << " ms" << std::endl;
    std::cout << "GPU time 1: " << t1.ms() << " ms" << std::endl;
    std::cout << "GPU time 2: " << t2.ms() << " ms" << std::endl;

    auto err = c - b;
    std::cout << "Max error 1: " << err.max() << std::endl;
    std::cout << "L2 error 2: " << err.norm() / err.size() << std::endl;

    auto err2 = d - c;
    std::cout << "Max error 2: " << err2.max() << std::endl;
    std::cout << "L2 error 2: " << err2.norm() / err2.size() << std::endl;

    return 0;
    // create
}
