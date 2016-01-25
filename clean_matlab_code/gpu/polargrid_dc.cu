#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
//#include "cutil.h"
#include "cuda.h"
#include <stdlib.h>
#include <cusp/complex.h>
#include <cusp/blas.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
//#include "mxGPUArray.h"
#include "polargrid.h"



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


void grid_points_gold(const float * d_point_pos_x, const float * d_point_pos_y, const cusp::complex<float>  * d_point_value,
		     const int npoints, const uint2 grid_size, const int * d_points_per_bin,const int * d_binned_points, 
		      const int * d_binned_points_idx, const int * d_bin_location, const int * d_bin_dim_x,
		      const int * d_bin_dim_y,int nbins, const float * d_kb_table, int kb_table_size,
		      float kb_table_scale,
		      cusp::complex<float> * d_grid_value){
  
  /* we're gonna receive almost all device pointers that we have to convert to CPU memory */

  float * point_pos_x = new float[npoints];
  cudaMemcpy(point_pos_x,d_point_pos_x,sizeof(float)*npoints,cudaMemcpyDeviceToHost);
  float * point_pos_y = new float[npoints];
  cudaMemcpy(point_pos_y,d_point_pos_y,sizeof(float)*npoints,cudaMemcpyDeviceToHost);

  cusp::complex<float> * point_value = new cusp::complex<float>[npoints];
  cudaMemcpy(point_value,d_point_value,sizeof(cusp::complex<float>)*npoints,cudaMemcpyDeviceToHost);
  int * points_per_bin = new int[nbins];
  cudaMemcpy(points_per_bin,d_points_per_bin,sizeof(int)*nbins,cudaMemcpyDeviceToHost);

  int * binned_points_idx = new int[nbins];
  cudaMemcpy(binned_points_idx,d_binned_points_idx,sizeof(int)*nbins,cudaMemcpyDeviceToHost);


  int total_size = 0;
  for(int i = 0;i<nbins;i++){
    total_size+= points_per_bin[i];
    total_size = 32*((total_size+31)/32);
  }  
  int * binned_points = new int[total_size];
  cudaMemcpy(binned_points,d_binned_points,sizeof(int)*total_size,cudaMemcpyDeviceToHost);

  int * bin_location = new int[nbins];
  cudaMemcpy(bin_location,d_bin_location,sizeof(int)*nbins,cudaMemcpyDeviceToHost);

  int * bin_dim_x = new int[nbins];
  cudaMemcpy(bin_dim_x,d_bin_dim_x,sizeof(int)*nbins,cudaMemcpyDeviceToHost);

  int * bin_dim_y = new int[nbins];
  cudaMemcpy(bin_dim_y,d_bin_dim_y,sizeof(int)*nbins,cudaMemcpyDeviceToHost);

  cusp::complex<float> * grid_value = new cusp::complex<float>[grid_size.x*grid_size.y];

  memset(grid_value,0,sizeof(cusp::complex<float>)*grid_size.x*grid_size.y);
  float * kb_table = new float[kb_table_size];
  cudaMemcpy(kb_table,d_kb_table,sizeof(float)*kb_table_size,cudaMemcpyDeviceToHost);
    
  for(int i = 0;i<nbins;i++){
    uint2 corner;
    corner.x = bin_location[i]%grid_size.x;
    corner.y = bin_location[i]/grid_size.x;
    int idx = binned_points_idx[i];
    for(int y = corner.y;y<corner.y+bin_dim_y[i];y++){
      for(int x = corner.x;x<corner.x+bin_dim_x[i];x++){
	grid_value[y*grid_size.x+x] = 0;
	for(int j = 0;j<points_per_bin[i];j++){
	  grid_value[y*grid_size.x+x] += point_value[binned_points[idx+j]]*
	    cpu_kb_weight(make_float2(x,y),
			  make_float2(point_pos_x[binned_points[idx+j]],
				      point_pos_y[binned_points[idx+j]]),
			  kb_table,
			  kb_table_size,
			  kb_table_scale);
	}
      }
    }
  }

  cudaMemcpy(d_grid_value,grid_value,sizeof(cusp::complex<float>)*grid_size.x*grid_size.y,cudaMemcpyHostToDevice);

}


#define	SX	    prhs[0]
#define	SY	prhs[1]
#define	SV	prhs[2]
#define	GRID_DIM	prhs[3]
#define	SPB	prhs[4]
#define	BIN_DIM_X      prhs[5]
#define	BIN_DIM_Y     prhs[6]
#define	SIB     prhs[7]
#define	BSO     prhs[8]
#define	BL     prhs[9]
#define	BPX     prhs[10]
#define	BPY     prhs[11]
#define	KLUT     prhs[12]
#define	KLUTS     prhs[13]


void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){

      /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    
 mxGPUArray const *samples_x;
 mxGPUArray const *samples_y;
 mxGPUArray const *samples_values;
 mxGPUArray const *samples_per_bin;
 mxGPUArray const *bin_dimensions_x;
 mxGPUArray const *bin_dimensions_y;
 mxGPUArray const *samples_in_bin;
 mxGPUArray const *bin_start_offset;
 mxGPUArray const *bin_location;
 mxGPUArray const *bin_points_x;
 mxGPUArray const *bin_points_y;
 mxGPUArray const *kernel_lookup_table;
//int *grid_dim =(int *) mxGetPr(GRID_DIM);
float kernel_lookup_table_scale = mxGetScalar(KLUTS);

 int *grid_dim0=( int *) (mxGetData(GRID_DIM));

//mwSize *grid_dim=(mwSize *)grid_dim0; 
mwSize *grid_dim={(mwSize *) mxGPUGetNumberOfElements(samples_x)};
//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim[0]),(grid_dim[1]));
//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim1[0]),(grid_dim1[1]));


// OUTPUT
mxGPUArray  *grid_values, *gold_grid_values;



samples_x = mxGPUCreateFromMxArray(SX);
samples_y = mxGPUCreateFromMxArray(SY);
samples_values = mxGPUCreateFromMxArray(SV);
samples_per_bin = mxGPUCreateFromMxArray(SPB);
bin_dimensions_x = mxGPUCreateFromMxArray(BIN_DIM_X);
bin_dimensions_y = mxGPUCreateFromMxArray(BIN_DIM_Y);  
samples_in_bin = mxGPUCreateFromMxArray(SIB);  
bin_start_offset = mxGPUCreateFromMxArray(BSO);  
bin_location = mxGPUCreateFromMxArray(BL);  
bin_points_x = mxGPUCreateFromMxArray(BPX); 
bin_points_y = mxGPUCreateFromMxArray(BPY); 
kernel_lookup_table= mxGPUCreateFromMxArray(KLUT); 

 int nbins = (int) (mxGPUGetNumberOfElements(bin_dimensions_x));
 int npoints =  (int)(mxGPUGetNumberOfElements(samples_x));
 int kernel_lookup_table_size = ( int)(mxGPUGetNumberOfElements(kernel_lookup_table));

mwSize ndim= 1;
// mwSize *grid_dim1[]={(mwSize grid_dim[0]), }
  
 
// output:
//  float2 * grid_values;
//  float2 * gold_grid_values;
 
//  plhs[0] = jkt_new( grid_dim[0], grid_dim[1], mxSINGLE_CLASS, mxREAL,);

//grid_values= mxGPUCreateGPUArray(ndim,grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
grid_values= mxGPUCreateGPUArray(ndim, grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

// now get the pointer or whatever it is
const float *d_samples_x = (const float  *)(mxGPUGetDataReadOnly(samples_x));
const  float *d_samples_y = (const float  *)(mxGPUGetDataReadOnly(samples_y));
// float2 *d_samples_values = (float2  *)(const float2  *)(mxGPUGetDataReadOnly(samples_values));
 const cusp::complex<float> *d_samples_values = (const cusp::complex<float> *)(mxGPUGetDataReadOnly(samples_values));
const  int * d_samples_per_bin = (const int  *)(mxGPUGetDataReadOnly(samples_per_bin));
const  int * d_bin_dimensions_x = (const int  *)(mxGPUGetDataReadOnly(bin_dimensions_x));
const int * d_bin_dimensions_y = (const int  *)(mxGPUGetDataReadOnly(bin_dimensions_y));
const  int * d_samples_in_bin = (const int  *)(mxGPUGetDataReadOnly(samples_in_bin));
const  int * d_bin_start_offset =(const int  *)(mxGPUGetDataReadOnly(bin_start_offset));
const  int * d_bin_location = (const int  *)(mxGPUGetDataReadOnly(bin_location));
const float * d_bin_points_x = (const float  *)(mxGPUGetDataReadOnly(bin_points_x));
const float * d_bin_points_y = (const float  *)(mxGPUGetDataReadOnly(bin_points_y));
const float * d_kernel_lookup_table = (const float  *)(mxGPUGetDataReadOnly(kernel_lookup_table));
const uint2 grid_size = {grid_dim[0],grid_dim[1]};

//float2 * d_grid_values = (float2  *)(mxGPUGetData(grid_values));
cusp::complex<float> * d_grid_values = (cusp::complex<float>  *)(mxGPUGetData(grid_values));


//   mexErrMsgTxt("gpuArray 1");

grid_points_cuda_interleaved_mex( d_samples_x, d_samples_y,
				    d_samples_values, npoints, 
				    grid_size, d_samples_per_bin, d_bin_dimensions_x, d_bin_dimensions_y,
				    d_samples_in_bin, d_bin_start_offset, d_bin_location, 
				    d_bin_points_x, d_bin_points_y,
				    nbins, d_kernel_lookup_table,
				    kernel_lookup_table_size,
				    kernel_lookup_table_scale,
				    d_grid_values);
//mexErrMsgTxt("gpuArray 2");



plhs[0] = mxGPUCreateMxArrayOnGPU(grid_values);


  if(nlhs == 2){
//gold_grid_values=  mxGPUCreateGPUArray(ndim, grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
gold_grid_values=  mxGPUCreateGPUArray(ndim, grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
//gold_grid_values= mxGPUCreateGPUArray(ndim,grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
//float2 * d_gold_grid_values = (float2  *)(mxGPUGetData(gold_grid_values));
cusp::complex<float> * d_gold_grid_values = ( cusp::complex<float> *)(mxGPUGetData(gold_grid_values));

    
  
 grid_points_gold (d_samples_x, d_samples_y,
 				    d_samples_values, npoints, 
 				    grid_size, d_samples_per_bin, d_samples_in_bin,   d_bin_start_offset, d_bin_location, 
                     d_bin_dimensions_x, d_bin_dimensions_y,nbins, d_kernel_lookup_table, kernel_lookup_table_size,
 				   kernel_lookup_table_scale,d_gold_grid_values);
	
        plhs[1] = mxGPUCreateMxArrayOnGPU(gold_grid_values);
	 mxGPUDestroyGPUArray( gold_grid_values);

  }
 mxGPUDestroyGPUArray( samples_x);
 mxGPUDestroyGPUArray( samples_y);
 mxGPUDestroyGPUArray( samples_values);
 mxGPUDestroyGPUArray( samples_per_bin);
 mxGPUDestroyGPUArray( bin_dimensions_x);
 mxGPUDestroyGPUArray( bin_dimensions_y);
 mxGPUDestroyGPUArray( samples_in_bin);
 mxGPUDestroyGPUArray( kernel_lookup_table);
 mxGPUDestroyGPUArray( bin_start_offset);
 mxGPUDestroyGPUArray( bin_location);
 mxGPUDestroyGPUArray( bin_points_x);
 mxGPUDestroyGPUArray( bin_points_y);
 mxGPUDestroyGPUArray( grid_values);

}


