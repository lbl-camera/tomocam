#ifndef _CUDA_SAMPLE_H_
#define _CUDA_SAMPLE_H_ 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <stdio.h>

/* constants */
const int BLOCKSIZE = 256;
const int GRIDSIZE = 4096 * 4;
const int SHARED_SIZE = 256;
const int SUM_SIZE = 256;


/* MACROS */
#define __cudafyit__ __device__ __inline__ static 

/* error handeling */
__inline__ void error_handle(cudaError_t status = cudaErrorLaunchFailure);
__inline__ void error_handle(cudaError_t status){
  if(status != cudaSuccess){
    cudaError_t s= cudaGetLastError();
    if(s != cudaSuccess){
      printf("%s\n",cudaGetErrorString(s));
      exit(1);
    }
  }
}


/* function overloads */

// multiply by scalar to the right
__cudafyit__ cuFloatComplex operator* (cuFloatComplex a, float b) {
    return make_cuFloatComplex(a.x * b, a.y * b);
}

// multiply scalar to the left
__cudafyit__ cuFloatComplex operator* (float a, cuFloatComplex b) {
    return make_cuFloatComplex(b.x * a, b.y * a);
}

// addition
__cudafyit__ cuFloatComplex operator + (cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(a, b);
}

// typedefs
typedef cuFloatComplex complex_t;
const complex_t CMPLX_ZERO = make_cuFloatComplex(0.f, 0.f);

// cuda calls

// TODO document
void polarsample(complex_t *, complex_t * , int , uint2 , float *, int, float, float, complex_t *);

// TODO document
void polarsample_transpose(complex_t *, complex_t * , int , uint2 , float *, int, float, float, complex_t *);


// TODO document
void addTVD(int , int, int, float, float, complex_t *, complex_t *);

// TODO document
void calcHessian(int , int, int, float, complex_t *, complex_t *);

#endif
