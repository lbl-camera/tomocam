#ifndef _CUDA_SAMPLE_H_
#define _CUDA_SAMPLE_H_ 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

/* typedefs */
typedef cuFloatComplex complex_t;

/* constants */
const int BLOCKSIZE = 256;
const int GRIDSIZE = 4096 * 4;
const int SHARED_SIZE = 256;
const int SUM_SIZE = 256;
const complex_t CMPLX_ZERO = make_cuFloatComplex(0.f, 0.f);


/* MACROS */
#define __cudafyit__ __device__ static __inline__ 

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
__cudafyit__ cuFloatComplex operator* (cuFloatComplex a, float b) {
    return make_cuFloatComplex(a.x * b, a.y * b);
}

__cudafyit__ cuFloatComplex operator += (cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

void polarsample(complex_t *, complex_t * , int , uint2 , float *, int, float, float, complex_t *);

void polarsample_transpose(complex_t *, complex_t * , int , uint2 , float *, int, float, float, complex_t *);


#endif
