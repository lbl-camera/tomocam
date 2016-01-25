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
#define	SGRID	prhs[2]
#define	GRID_DIM	prhs[3]
#define	KLUT     prhs[4]
#define	KLUTS     prhs[5]
#define	KR     prhs[6]


void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
      /* Initialize the MathWorks GPU API. */
    mxInitGPU();

 mxGPUArray const *samples_x;
 mxGPUArray const *samples_y;
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


// 
samples_x = mxGPUCreateFromMxArray(SX);
samples_y = mxGPUCreateFromMxArray(SY);
grid_values = mxGPUCreateFromMxArray(SGRID);
 kernel_lookup_table=mxGPUCreateFromMxArray(KLUT); 

//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim[0]),(grid_dim[1]));
//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim1[0]),(grid_dim1[1]));
//mexPrintf("-\n");
// 

int npoints =  (int)(mxGPUGetNumberOfElements(samples_x));
int kernel_lookup_table_size = (int)(mxGPUGetNumberOfElements(kernel_lookup_table));
// mwSize ndim= 1;
//mwSize const grid_dim2[]={npoints};
  
mwSize const *sample_dim= mxGPUGetDimensions(samples_x);
mwSize ndim= mxGPUGetNumberOfDimensions(samples_x);
//plhs[0]  =mxCreateNumericArray(ndim,grid_dim2,mxSINGLE_CLASS,mxCOMPLEX);

//mexErrMsgTxt("gpuArray 0");

samples_values= mxGPUCreateGPUArray(ndim,sample_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

//samples_values= mxGPUCreateGPUArray(ndim,grid_dim2,mxINT32_CLASS,mxREAL, MX_GPU_INITIALIZE_VALUES);
//samples_values= 0;

//POINTERS
float *d_samples_x = (float *)(const float  *)(mxGPUGetDataReadOnly(samples_x));
 float *d_samples_y = (float  *)(const float  *)(mxGPUGetDataReadOnly(samples_y));
cusp::complex<float> * d_grid_values = (cusp::complex<float>  *)(const cusp::complex<float>  *)(mxGPUGetDataReadOnly(grid_values));
float * d_kernel_lookup_table = (float *)(const float  *)(mxGPUGetDataReadOnly(kernel_lookup_table));


uint2 grid_size = {grid_dim[0],grid_dim[1]};


//mexErrMsgTxt("gpuArray 0");
// OUTPUT
//float2 * d_samples_values = (float2  *)(mxGPUGetData(samples_values));
cusp::complex<float> * d_samples_values = ( cusp::complex<float> *)(mxGPUGetData(samples_values));

// mexErrMsgTxt("gpuArray 1");
 
 
 cuda_sample( d_samples_x, d_samples_y,
	       d_grid_values, npoints, 
	       grid_size, 
 	       d_kernel_lookup_table,
	       kernel_lookup_table_size,
	       kernel_lookup_table_scale,
           kernel_radius,
	       d_samples_values);  

  // GET OUTPUT
  plhs[0] = mxGPUCreateMxArrayOnGPU(samples_values);

// compute gold
  if(nlhs >= 2){
mxGPUArray  *samples_values_gold;
samples_values_gold= mxGPUCreateGPUArray(ndim,sample_dim,mxSINGLE_CLASS,mxCOMPLEX,MX_GPU_DO_NOT_INITIALIZE);
   cusp::complex<float> * d_samples_values_gold = (cusp::complex<float>  *)(mxGPUGetData(samples_values_gold));

    sample_points_gold(d_samples_x, d_samples_y, d_grid_values,
		       npoints, grid_size,d_kernel_lookup_table, kernel_lookup_table_size,
		       kernel_lookup_table_scale, kernel_radius,
		       d_samples_values_gold);
  plhs[1] = mxGPUCreateMxArrayOnGPU(samples_values_gold);
    mxGPUDestroyGPUArray(samples_values_gold);
}


 mxGPUDestroyGPUArray( samples_x);
 mxGPUDestroyGPUArray( samples_y);
 mxGPUDestroyGPUArray( grid_values);
 mxGPUDestroyGPUArray( kernel_lookup_table);
 mxGPUDestroyGPUArray(samples_values);

}
