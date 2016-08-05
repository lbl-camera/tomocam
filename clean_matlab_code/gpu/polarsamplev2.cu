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
#define	SGRID	prhs[1]
#define	GRID_DIM	prhs[2]
#define	KLUT     prhs[3]
#define	KLUTS     prhs[4]
#define	KR     prhs[5]
#define	KB     prhs[6]


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


__global__ void cuda_sample_kernel(cusp::complex<float> * point_pos,
				   cusp::complex<float> * grid_value, 
				   int npoints, uint2 grid_size,
				   int kb_table_size,
				   float kb_table_scale,
				   float kernel_radius,
				   float kernel_beta,
				    cusp::complex<float> * sample_value){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i < npoints){
    cusp::complex<float> sv=0;
    cusp::complex<float> pp=point_pos[i];
    float sx=pp.real();
    float sy=pp.imag();

    for(int y = max(0.0f,ceil(sy-kernel_radius));y<= min(floor(sy+kernel_radius),grid_size.y-1.0f);y++){
      if(y < 0 || y > grid_size.y-1){
	continue;
      }
      float  kby=kb_weight(y,sy,kb_table_size,kb_table_scale);
      //float  kby=kb_weight_a(y,sy,kernel_radius,kernel_beta);
 
      for(int x = max(0.0f,ceil(sx-kernel_radius));x<= min(floor(sx+kernel_radius),grid_size.x-1.0f);x++){
	if(x < 0 || x > grid_size.x-1){
	  continue;
	}
	sv += grid_value[y*grid_size.x+x]*kby*kb_weight(x,sx,kb_table_size, kb_table_scale);
	//sv += grid_value[y*grid_size.x+x]*kby*kb_weight_a(x,sx,kernel_radius, kernel_beta);
	
      }
    }  
    sample_value[i] = sv;

  }
}

void cuda_sample(cusp::complex<float> * point_pos,
		 cusp::complex<float> * grid_value, int npoints, 
		 uint2 grid_size,
		 float * kb_table,
		 int kb_table_size,
		 float kb_table_scale,
		 float kernel_radius, float kernel_beta,		 
		 cusp::complex<float> * sample_value){
  cudaMemset(sample_value,0,sizeof( cusp::complex<float>)*npoints);

  size_t offset;
  cudaBindTexture(&offset,texRef, kb_table, sizeof(float)*kb_table_size);
  if(offset != 0){
    printf("Error: Texture offset different than zero. Table not allocated with cudaMalloc!%d\n");
    return;
  }

  int block_size = BLOCKSIZE;
  int grid = (npoints+block_size-1)/block_size;
  clock_t t_i = clock();
    cuda_sample_kernel<<<grid,block_size>>>( point_pos,
					     grid_value, npoints, 
					     grid_size,
					     kb_table_size,
					     kb_table_scale,
					     kernel_radius,kernel_beta,
					     sample_value);
    cudaThreadSynchronize();
    

  clock_t t_e = clock();
  error_handle();
  //  printf("%d iter in %5.1f ms\n",iter,(t_e-t_i)*1000.0/CLOCKS_PER_SEC);
}


void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
      /* Initialize the MathWorks GPU API. */
    mxInitGPU();

 mxGPUArray const *samples_pos;
 mxGPUArray const *grid_values;
 mxGPUArray const *kernel_lookup_table;
// OUTPUT
mxGPUArray  *samples_values;

//double *grid_dim = mxGetPr(GRID_DIM);
//int *grid_dim1 =(int *) mxGetPr(GRID_DIM);
 int *grid_dim0=( int *) (mxGetData(GRID_DIM));
//mwSize const *grid_dim =(mwSize const *) mxGetPr(GRID_DIM);
// mwSize *grid_dim =(mwSize *) *grid_dim1;

mwSize *grid_dim=(mwSize *)grid_dim0; 

float kernel_lookup_table_scale = mxGetScalar(KLUTS);
float  kernel_radius = mxGetScalar(KR);
float  kernel_beta = mxGetScalar(KB);


// 
samples_pos = mxGPUCreateFromMxArray(SXY);
grid_values = mxGPUCreateFromMxArray(SGRID);
 kernel_lookup_table=mxGPUCreateFromMxArray(KLUT); 

//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim[0]),(grid_dim[1]));
//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim1[0]),(grid_dim1[1]));
//mexPrintf("-\n");
// 

int npoints =  (int)(mxGPUGetNumberOfElements(samples_pos));
int kernel_lookup_table_size = (int)(mxGPUGetNumberOfElements(kernel_lookup_table));
// mwSize ndim= 1;
//mwSize const grid_dim2[]={npoints};
  
mwSize const *sample_dim= mxGPUGetDimensions(samples_pos);
mwSize ndim= mxGPUGetNumberOfDimensions(samples_pos);
//plhs[0]  =mxCreateNumericArray(ndim,grid_dim2,mxSINGLE_CLASS,mxCOMPLEX);

//mexErrMsgTxt("gpuArray 0");

samples_values= mxGPUCreateGPUArray(ndim,sample_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

//samples_values= mxGPUCreateGPUArray(ndim,grid_dim2,mxINT32_CLASS,mxREAL, MX_GPU_INITIALIZE_VALUES);
//samples_values= 0;

//POINTERS

cusp::complex<float> * d_samples_pos = (cusp::complex<float>  *)(const cusp::complex<float>  *)(mxGPUGetDataReadOnly(samples_pos));
cusp::complex<float> * d_grid_values = (cusp::complex<float>  *)(const cusp::complex<float>  *)(mxGPUGetDataReadOnly(grid_values));
float * d_kernel_lookup_table = (float *)(const float  *)(mxGPUGetDataReadOnly(kernel_lookup_table));

uint2 grid_size = {grid_dim[0],grid_dim[1]};


// OUTPUT
cusp::complex<float> * d_samples_values = ( cusp::complex<float> *)(mxGPUGetData(samples_values));

 cuda_sample( d_samples_pos,
	       d_grid_values, npoints, 
	       grid_size, 
 	       d_kernel_lookup_table,
	       kernel_lookup_table_size,
	       kernel_lookup_table_scale,
	      kernel_radius, kernel_beta,
	       d_samples_values);  

  // GET OUTPUT
  plhs[0] = mxGPUCreateMxArrayOnGPU(samples_values);



 mxGPUDestroyGPUArray( samples_pos);
 mxGPUDestroyGPUArray( grid_values);
 mxGPUDestroyGPUArray( kernel_lookup_table);
 mxGPUDestroyGPUArray(samples_values);

}
