#ifndef TOMOCAM_DEBUG__H
#define TOMOCAM_DEBUG__H

#include <fstream>
#include <cuda_runtime.h>
#include "common.h"

namespace tomocam {
    inline void write_output(cuComplex_t * data, dim3_t dims) {

        size_t SIZE = dims.y * dims.z;
        float * real = new float[SIZE];
        float * imag = new float[SIZE];

        float * tmp = (float *) data;
        cudaMemcpy2D(real, sizeof(float), tmp, sizeof(cuComplex_t), sizeof(float), SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(imag, sizeof(float), tmp+1, sizeof(cuComplex_t), sizeof(float), SIZE, cudaMemcpyDeviceToHost);

        std::ofstream fp("real.out", std::fstream::out);
        fp.write((char *) real, SIZE * sizeof(float));
        fp.close();
    
        std::ofstream fq("imag.out", std::fstream::out);
        fq.write((char *) imag, SIZE * sizeof(float));
        fq.close();
    
        exit(1);
    }
}

#endif // TOMOCAM_DEBUG__H

