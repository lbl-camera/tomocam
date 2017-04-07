
#include <cuda.h>
#include <stdio.h>

#include "polarsample.h"

int main(int argc, char ** argv){

    if (argc != 4){
        printf("Usage: %s <nslice> <nrow> <ncol> \n", argv[0]);
        exit(1);
    }
    int nslice = atoi(argv[1]);
    int nrow = atoi(argv[2]);
    int ncol = atoi(argv[3]);


    size_t n = nslice * nrow * ncol;
    complex_t * f = new complex_t[n];
    complex_t * v  = new complex_t[n];

    for (int i = 0; i < n; i++){
        v[i] = make_cuFloatComplex(1.f, 0.f);
        f[i] = make_cuFloatComplex(0.f, 0.f);
    }

    complex_t * d_f = NULL;
    complex_t * d_v = NULL;

    cudaMalloc((void **) &d_f, n * sizeof(complex_t));
    cudaMalloc((void **) &d_v, n * sizeof(complex_t));

    cudaMemcpy(d_v, v, n * sizeof(complex_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, n * sizeof(complex_t), cudaMemcpyHostToDevice);

    addTVD(nslice, nrow, ncol, d_f, d_v);

    delete []f;
    delete []v;

    cudaFree(d_v);
    cudaFree(d_f);
    return 0;
} 
