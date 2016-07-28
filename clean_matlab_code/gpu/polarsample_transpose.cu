#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cusp/complex.h>
#include "polargrid.h"
#include "cuda_sample.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"


#define	SX	    prhs[0]
#define	SY	prhs[1]
#define	SV     prhs[2]
#define	GRID_DIM	prhs[3]
#define	KLUT     prhs[4]
#define	KLUTS     prhs[5]
#define	KR     prhs[6]

#define	SGRID	plhs[0]

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



__global__ void cuda_sample_transpose_kernel(const float * point_pos_x,
					     const float * point_pos_y,
					     const cusp::complex<float> * sample_value, 
					     int npoints, uint2 grid_size,
					     int kb_table_size,
					     float kb_table_scale,
					     float kernel_radius,
					     cusp::complex<float> * grid_value){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i < npoints){

    cusp::complex<float> sv=  sample_value[i] ;
    float sx=point_pos_x[i];
    float sy=point_pos_y[i];

    for(int y = max(0.0f,ceil(sy-kernel_radius));y<= min(floor(sy+kernel_radius),grid_size.y-1.0f);y++){
      if(y < 0 || y > grid_size.y-1){
	continue;
      }
      //sv=sv*kb_weight(y,sy,kb_table_size,kb_table_scale);

      for(int x = max(0.0f,ceil(sx-kernel_radius));x<= min(floor(sx+kernel_radius),grid_size.x-1.0f);x++){
	if(x < 0 || x > grid_size.x-1){
	  continue;
	}
	//	grid_value[y*grid_size.x+x]+=	  sv * 	  kb_weight(x,sx, kb_table_size, kb_table_scale);	
	grid_value[y*grid_size.x+x]+=	  sv * 	  kb_weight(make_float2(x,y),   make_float2(sx,sy), kb_table_size, kb_table_scale);	

      }
    }

  }
}

void cuda_sample_transpose(const float * point_pos_x, const float * point_pos_y,
		 const cusp::complex<float> * sample_value, int npoints, 
		 uint2 grid_size,
		 const float * kb_table,
		 int kb_table_size,
		 float kb_table_scale,
		 float kernel_radius,		 
		 cusp::complex<float> * grid_value){
  //  cudaMemset(sample_value,0,sizeof( cusp::complex<float>)*npoints);

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
    cuda_sample_transpose_kernel<<<grid,block_size>>>( point_pos_x, point_pos_y,
					     sample_value, npoints, 
					     grid_size,
					     kb_table_size,
					     kb_table_scale,
					     kernel_radius,
					     grid_value);
    cudaThreadSynchronize();
    
  }
  clock_t t_e = clock();
  error_handle();
  //  printf("%d iter in %5.1f ms\n",iter,(t_e-t_i)*1000.0/CLOCKS_PER_SEC);
}

void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
      /* Initialize the MathWorks GPU API. */
    mxInitGPU();

 mxGPUArray const *samples_x;
 mxGPUArray const *samples_y;
 //OUTPUT
 // mxGPUArray const *grid_values;
 mxGPUArray *grid_values;

 mxGPUArray const *kernel_lookup_table;
// INPUT
// mxGPUArray  *samples_values;
 mxGPUArray const *samples_values;

 int *grid_dim0=( int *) (mxGetData(GRID_DIM));
mwSize *grid_dim=(mwSize *)grid_dim0; 

float kernel_lookup_table_scale = mxGetScalar(KLUTS);
float  kernel_radius = mxGetScalar(KR);

// 
samples_x = mxGPUCreateFromMxArray(SX);
samples_y = mxGPUCreateFromMxArray(SY);
kernel_lookup_table=mxGPUCreateFromMxArray(KLUT); 

int npoints =  (int)(mxGPUGetNumberOfElements(samples_x));
int kernel_lookup_table_size = (int)(mxGPUGetNumberOfElements(kernel_lookup_table));

 
mwSize const *sample_dim= mxGPUGetDimensions(samples_x);
mwSize ndim= mxGPUGetNumberOfDimensions(samples_x);

// grid_values = mxGPUCreateFromMxArray(SGRID);
// samples_values= mxGPUCreateGPUArray(ndim,sample_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

samples_values= mxGPUCreateFromMxArray(SV);

grid_values = mxGPUCreateGPUArray(ndim,grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);

//samples_values= mxGPUCreateGPUArray(ndim,grid_dim2,mxINT32_CLASS,mxREAL, MX_GPU_INITIALIZE_VALUES);
//samples_values= 0;

//POINTERS
const float *d_samples_x = (const float  *)(mxGPUGetDataReadOnly(samples_x));
const  float *d_samples_y = (const float  *)(mxGPUGetDataReadOnly(samples_y));
const float * d_kernel_lookup_table = (const float  *)(mxGPUGetDataReadOnly(kernel_lookup_table));

uint2 grid_size = {grid_dim[0],grid_dim[1]};


//cusp::complex<float> * d_samples_values = ( cusp::complex<float> *)(mxGPUGetData(samples_values));
const cusp::complex<float> * d_samples_values = (const cusp::complex<float>  *)(mxGPUGetDataReadOnly(samples_values));


// OUTPUT
// cusp::complex<float> * d_grid_values = (cusp::complex<float>  *)(const cusp::complex<float>  *)(mxGPUGetDataReadOnly(grid_values));
cusp::complex<float> * d_grid_values = (cusp::complex<float>  *)(mxGPUGetData(grid_values));

// mexErrMsgTxt("gpuArray 1");
 
 cuda_sample_transpose( d_samples_x, d_samples_y,
			d_samples_values,
			npoints, 
			grid_size, 
			d_kernel_lookup_table,
			kernel_lookup_table_size,
			kernel_lookup_table_scale,
			kernel_radius,
			d_grid_values    );  

  // GET OUTPUT
 //  plhs[0] = mxGPUCreateMxArrayOnGPU(samples_values);

  plhs[0] = mxGPUCreateMxArrayOnGPU(grid_values);

 mxGPUDestroyGPUArray( samples_x);
 mxGPUDestroyGPUArray( samples_y);
 mxGPUDestroyGPUArray( grid_values);
 mxGPUDestroyGPUArray( kernel_lookup_table);

 // mxGPUDestroyGPUArray( grid_values);
 mxGPUDestroyGPUArray(samples_values);

}