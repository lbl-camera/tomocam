#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cusp/complex.h>
#include "polargrid.h"
#include "cuda_sample.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"


#define	SXY	    prhs[0]
#define	KLUT     prhs[1]
#define	KLUTS     prhs[2]
#define	KR     prhs[3]
#define	KB     prhs[4]


texture<float, 1, cudaReadModeElementType> texRef;

void error_handle(cudaError_t status = cudaErrorLaunchFailure);

void error_handle(cudaError_t status){
  if(status != cudaSuccess){
    cudaError_t s= cudaGetLastError();
    if(s != cudaSuccess){
      printf("%s\n",cudaGetErrorString(s));
      exit(1);
    }
  }
}

float cpu_kb_weight(float2 grid_pos, float2 point_pos,
		    float * kb_table,
		    int kb_table_size,
		    float kb_table_scale){
  float dist_x = fabs(grid_pos.x-point_pos.x)*kb_table_scale;
  float dist_y = fabs(grid_pos.y-point_pos.y)*kb_table_scale;
  int ix = (int)dist_x;
  float fx = dist_x-ix;
  int iy = (int)dist_y;
  float fy = dist_y-iy;

  if(ix+1 < kb_table_size && iy+1 < kb_table_size){
    return (kb_table[ix]*(1.0f-fx) + kb_table[ix+1]*(fx)) *
      (kb_table[iy]*(1.0f-fy) + kb_table[iy+1]*(fy));     
  }
  return 0.0f;
}


__device__ float kb_weight(float grid_pos, float point_pos,
				    int kb_table_size,
				    float kb_table_scale){
  float dist_x = fabs(grid_pos-point_pos)*kb_table_scale;

  int ix = (int)dist_x;
  float fx = dist_x-ix;
  if(ix+1 < kb_table_size){
  return (tex1Dfetch(texRef,ix)*(1.0f-fx) + tex1Dfetch(texRef,ix+1)*(fx));     
  }
  return 0.0f;
}


__device__ float kb_weight_a(float grid_pos, float point_pos,
				    int k_r,   float kbeta){
  float dist_x = fabs(grid_pos-point_pos);
 
  if(dist_x<k_r){
    dist_x*=2/k_r;
    dist_x*=dist_x;
  return     cyl_bessel_i0f(kbeta* sqrtf( 1-dist_x ));
  }
  return 0.0f;
}


__device__ float kb_weight(float2 grid_pos, float2 point_pos,
				    int kb_table_size,
				    float kb_table_scale){
  float dist_x = fabs(grid_pos.x-point_pos.x)*kb_table_scale;
  float dist_y = fabs(grid_pos.y-point_pos.y)*kb_table_scale;

  int ix = (int)dist_x;
  float fx = dist_x-ix;
  int iy = (int)dist_y;
  float fy = dist_y-iy;

  if(ix+1 < kb_table_size && iy+1 < kb_table_size){
  return (tex1Dfetch(texRef,ix)*(1.0f-fx) + tex1Dfetch(texRef,ix+1)*(fx)) * 
    (tex1Dfetch(texRef,iy)*(1.0f-fy) + tex1Dfetch(texRef,iy+1)*(fy));     
  }
  return 0.0f;
}


__global__ void kb_sample_kernel(float * point_pos,
				   int npoints,
				   int kb_table_size,
				   float kb_table_scale,
				   float kernel_radius,
				   float kernel_beta,
				   float * sample_value){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i < npoints){
    cusp::complex<float> sv=0;
    float sx=point_pos[i];
    float x=0;

    // float  kby=kb_weight_a(y,sy,kernel_radius,kernel_beta);
    //float kbx =kb_weight(x,sx,kb_table_size, kb_table_scale);
    float  kbx= kb_weight_a(x,sx,kernel_radius, kernel_beta);	
    sample_value[i] = kbx;
  }
}

void kb_sample(float * point_pos,
		 int npoints, 
		 float * kb_table,
		 int kb_table_size,
		 float kb_table_scale,
		 float kernel_radius, float kernel_beta,		 
		 float * sample_value){

  size_t offset;
  cudaBindTexture(&offset,texRef, kb_table, sizeof(float)*kb_table_size);
  if(offset != 0){
    printf("Error: Texture offset different than zero. Table not allocated with cudaMalloc!%d\n");
    return;
  }

  int block_size = BLOCKSIZE;
  int grid = (npoints+block_size-1)/block_size;
  clock_t t_i = clock();
  int iter = 1;
  for(int i = 0;i<iter;i++){
    kb_sample_kernel<<<grid,block_size>>>( point_pos,
					     npoints, 
					     kb_table_size,
					     kb_table_scale,
					     kernel_radius,kernel_beta,
					     sample_value);
    cudaThreadSynchronize();
    
  }
  clock_t t_e = clock();
  error_handle();
  //  printf("%d iter in %5.1f ms\n",iter,(t_e-t_i)*1000.0/CLOCKS_PER_SEC);
}


void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
      /* Initialize the MathWorks GPU API. */
    mxInitGPU();

 mxGPUArray const *samples_pos;
 mxGPUArray const *kernel_lookup_table;
// OUTPUT
mxGPUArray  *samples_values;

float kernel_lookup_table_scale = mxGetScalar(KLUTS);
float  kernel_radius = mxGetScalar(KR);
float  kernel_beta = mxGetScalar(KB);


samples_pos = mxGPUCreateFromMxArray(SXY);

 kernel_lookup_table=mxGPUCreateFromMxArray(KLUT); 


int npoints =  (int)(mxGPUGetNumberOfElements(samples_pos));
int kernel_lookup_table_size = (int)(mxGPUGetNumberOfElements(kernel_lookup_table));
  
mwSize const *sample_dim= mxGPUGetDimensions(samples_pos);
mwSize ndim= mxGPUGetNumberOfDimensions(samples_pos);


samples_values= mxGPUCreateGPUArray(ndim,sample_dim,mxSINGLE_CLASS,mxREAL, MX_GPU_DO_NOT_INITIALIZE);

float * d_samples_pos = (float  *)(const float  *)(mxGPUGetDataReadOnly(samples_pos));
float * d_kernel_lookup_table = (float *)(const float  *)(mxGPUGetDataReadOnly(kernel_lookup_table));

// OUTPUT
float * d_samples_values = ( float *)(mxGPUGetData(samples_values));

kb_sample( d_samples_pos,
	       npoints, 
	       d_kernel_lookup_table,
	       kernel_lookup_table_size,
	       kernel_lookup_table_scale,
	       kernel_radius, kernel_beta,
	       d_samples_values);  

  // GET OUTPUT
  plhs[0] = mxGPUCreateMxArrayOnGPU(samples_values);



 mxGPUDestroyGPUArray( samples_pos);
 mxGPUDestroyGPUArray( kernel_lookup_table);
 mxGPUDestroyGPUArray(samples_values);

}
