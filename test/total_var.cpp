#include <algorithm>
#include <cmath>
#include <iostream>

#include "dist_array.h"
#include "test_utils.h"
#include "tomocam.h"

namespace {

    const float weight[3][3][3] = {
        {{0.0302f, 0.037f, 0.0302f}, {0.037f, 0.0523f, 0.037f},
            {0.0302f, 0.037f, 0.0302f}},
        {{0.037f, 0.0523f, 0.037f}, {0.0523f, 0.f, 0.0523f},
            {0.037f, 0.0523f, 0.037f}},
        {{0.0302f, 0.037f, 0.0302f}, {0.037f, 0.0523f, 0.037f},
            {0.0302f, 0.037f, 0.0302f}}};

    const float MRF_Q = 2.f;
    const float MRF_C = 0.001f;

    float d_potfun(float delta, float sigma, float p) {
        float sigma_q = std::pow(sigma, MRF_Q);
        float sigma_q_p = std::pow(sigma, MRF_Q - p);
        float temp1 = std::pow(std::abs(delta), MRF_Q - p) / sigma_q_p;
        float temp2 = std::pow(std::abs(delta), MRF_Q - 1);
        float temp3 = MRF_C + temp1;

        if (delta > 0.f)
            return (temp2 / (temp3 * sigma_q)) *
                   (MRF_Q - ((MRF_Q - p) * temp1) / temp3);
        else if (delta < 0.f)
            return (-1.f * temp2 / (temp3 * sigma_q)) *
                   (MRF_Q - ((MRF_Q - p) * temp1) / temp3);
        else
            return 0.f;
    }

    // CPU reference for add_total_var2's qGGMRF TV-penalty gradient term --
    // the term nagopt's gradient/loss calc depends on (src/tv_update.cpp).
    void cpu_total_var(tomocam::DArray<float> &input,
        tomocam::DArray<float> &output, float sigma, float mrf_p) {
        tomocam::dim3_t dims = output.dims();
        int nslc = dims.x, nrow = dims.y, ncol = dims.z;
        for (int i = 0; i < nslc; i++) {
            for (int j = 0; j < nrow; j++) {
                for (int k = 0; k < ncol; k++) {
                    float u = input(i, j, k);
                    float v = 0.f;
                    for (int z = 0; z < 3; z++) {
                        for (int y = 0; y < 3; y++) {
                            for (int x = 0; x < 3; x++) {
                                int i1 = std::clamp(i + z - 1, 0, nslc - 1);
                                int j1 = std::clamp(j + y - 1, 0, nrow - 1);
                                int k1 = std::clamp(k + x - 1, 0, ncol - 1);
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

} // namespace

int main() {
    float p = 1.2f, sigma = 10.f;

    tomocam::dim3_t dims(2, 33, 33); // small, non-power-of-2 boundary exercise
    auto a = random_uniform<float>(dims, 0.f, 1.f, 8);
    tomocam::DArray<float> b(dims), c(dims);
    b.init(0.f);
    c.init(0.f);

    cpu_total_var(a, b, sigma, p);
    tomocam::add_total_var2(a, c, sigma, p);

    double max_err = 0;
    for (uint64_t i = 0; i < b.size(); i++)
        max_err = std::max(max_err, static_cast<double>(std::abs(b[i] - c[i])));

    std::cout << "max |cpu - gpu| add_total_var2 error: " << max_err << std::endl;

    bool ok = max_err < 1e-3;
    return report(ok);
}
