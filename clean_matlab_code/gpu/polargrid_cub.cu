#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cusp/complex.h>
#include <cusp/blas.h>
#include <cub/cub.cuh>
#include <thrust/reduce.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"
#include "polargrid.h"
//#include "stdafx.h"
#include "cuda_profiler_api.h"

texture<int,1> tex_x_int;
texture<float,1> tex_x_float;
texture<float,1> tex_x_float1;


__inline__ __device__ cusp::complex<float> fetch_x(const int& i, const cusp::complex<float> * x)
{
return cusp::complex<float>(tex1Dfetch(tex_x_float, i*2),tex1Dfetch(tex_x_float, i*2+1));
}

__inline__ __device__ float fetch_x(const int& i,const float * x)
{
return tex1Dfetch(tex_x_float1, i);
}

__inline__ __device__ cusp::complex<float> fetch_xc(const int& i, const cusp::complex<float> * x)
{
return cusp::complex<float>(tex1Dfetch(tex_x_float, i*2),tex1Dfetch(tex_x_float, i*2+1));
}

__inline__ __device__ cusp::complex<float> shflb( const cusp::complex<float>  x, int& i)
{
  //  return cusp::complex<float>(__shfl(float2(x),i));
  //  return cusp::complex<float>(__shfl(float(x),i*2),(__shfl(float(x),i*2+1)));
  return cusp::complex<float>(__shfl(x.x,i),(__shfl(x.y,i)));
}

__inline__ __device__ float shflb( const float  x, int& i)
{
  return float(__shfl(x,i));
}

__inline__ __device__ void atomicAdd( cusp::complex<float> * x,  cusp::complex<float>  m)
{
  //  cusp::complex<float> m;
     atomicAdd(&(x[0]).x,m.x);
     atomicAdd(&(x[0]).y,m.y);
      //      return m;
}



void error_handle(cudaError_t status = cudaErrorLaunchFailure);

void error_handle(cudaError_t status){
    if(status != cudaSuccess){
        cudaError_t s= cudaGetLastError();
        if(s != cudaSuccess){
            //  printf("%s\n",cudaGetErrorString(s));
            exit(1);
        }
    }
}



__device__ float kb_weight(float2 grid_pos, float2 point_pos,
        int kb_table_size, float kb_table_scale,                const float * kb_table){
    float dist_x = fabsf(grid_pos.x-point_pos.x)*kb_table_scale;
    float dist_y = fabsf(grid_pos.y-point_pos.y)*kb_table_scale;
    //float w=tex1D<float>(texRef,0.0f);

    //    return w;//tex1D<float>(texRef,dist_y);// *tex1D<float>(texRef,dist_y);

//      return 1.0f;
    
    int ix = (int)dist_x;
    float fx = dist_x-ix;
    int iy = (int)dist_y;
    float fy = dist_y-iy;
    
    if(ix+1 < kb_table_size && iy+1 < kb_table_size){
      //  return (tex1Dfetch<float>(texRef,ix)*(1.0f-fx) + tex1Dfetch<float>(texRef,ix+1)*(fx)) *
      //           (tex1Dfetch<float>(texRef,iy)*(1.0f-fy) + tex1Dfetch<float>(texRef,iy+1)*(fy));

return	(fetch_x(ix,kb_table)*(1.0f-fx)+ fetch_x(ix+1,kb_table)*(fx))* 
	(fetch_x(iy,kb_table)*(1.0f-fy)+ fetch_x(iy+1,kb_table)*(fy)); 
    }
    return 0.0f;
    /*    */
}

__device__ float kb_weight(float grid_x, float grid_y, float point_pos_x,
        float point_pos_y,
        int kb_table_size,
    float kb_table_scale, const float * kb_table){
    float dist_x = fabsf(grid_x-point_pos_x)*kb_table_scale;
    float dist_y = fabsf(grid_y-point_pos_y)*kb_table_scale;
    //    return tex1D<float>(texRef,dist_x) *tex1D<float>(texRef,dist_y);


     int ix = (int)dist_x;
    float fx = dist_x-ix;
    int iy = (int)dist_y;
    float fy = dist_y-iy;
    
    if(ix+1 < kb_table_size && iy+1 < kb_table_size){
return	(fetch_x(ix,kb_table)*(1.0f-fx)+ fetch_x(ix+1,kb_table)*(fx))* 
	(fetch_x(iy,kb_table)*(1.0f-fy)+ fetch_x(iy+1,kb_table)*(fy)); 
/*
        return (tex1Dfetch<float>(texRef,ix)*(1.0f-fx) + tex1Dfetch<float>(texRef,ix+1)*(fx)) *
                (tex1Dfetch<float>(texRef,iy)*(1.0f-fy) + tex1Dfetch<float>(texRef,iy+1)*(fy));
*/

    }
    return 0.0f;

}

__device__ float kb_weight(float2 grid_pos, float2 point_pos,
        int kb_table_size,
			   float kb_table_scale,                const float * kb_table,int tid){
    float dist_x = fabsf(grid_pos.x-point_pos.x)*kb_table_scale;
    float dist_y = fabsf(grid_pos.y-point_pos.y)*kb_table_scale;
    //  return 0.0f;
    //    return tex1D<float>(texRef,dist_x) *tex1D<float>(texRef,dist_y);


    //    return tex1D<float>(texRef,dist_x) *tex1D<float>(texRef,dist_y);

  
    float ix = rintf(dist_x);
    float fx = dist_x-ix;
    float iy = rintf(dist_y);
    float fy = dist_y-iy;
    
    if(ix+1 < kb_table_size && iy+1 < kb_table_size){

return	(fetch_x(ix,kb_table)*(1.0f-fx)+ fetch_x(ix+1,kb_table)*(fx))* 
	(fetch_x(iy,kb_table)*(1.0f-fy)+ fetch_x(iy+1,kb_table)*(fy)); 

/*
        return (tex1Dfetch<float>(texRef,tid)*(1.0f-fx) + tex1Dfetch<float>(texRef,tid)*(fx)) *
                (tex1Dfetch<float>(texRef,tid)*(1.0f-fy) + tex1Dfetch<float>(texRef,tid)*(fy));
*/
    }
    return 0.0f;
  

}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void sum_points(        const cusp::complex<float> * point_value,
        int npoints,  uint2 grid_size,
        const int *  points_per_bin,
        const int * bin_dimension_x,
        const int * bin_dimension_y,
        const int *  binned_points,
        const int * binned_points_idx,
        const int * bin_location,
        const float * binned_points_x,
        const float * binned_points_y,
        const int nbins,
        const int kb_table_size,
        const float kb_table_scale,                const float * kb_table, cudaTextureObject_t texRef,
			   cusp::complex<float> * grid_value,int pbid){
  __shared__ cusp::complex<float> value;
  
  __shared__ cusp::complex<float> sum_t[BLOCKSIZE];
    
  // Specialize BlockReduce for a 1D block of 128 threads on type cusp::complex<float>
  
  //    int i = blockIdx.x;

  int i = pbid;
  int tid = threadIdx.x;
  
  int jj = blockIdx.x;
  
  uint2 corner;
  corner.x = bin_location[i]%grid_size.x;
  corner.y = bin_location[i]/grid_size.x;
  const int idx = binned_points_idx[i];
  const int ppb = points_per_bin[i];
  //    cusp::complex<float> * value;
  const int  bd=BLOCKSIZE;
  //	const int  bd=blockDim.x;
  //const uint2 dims = {bin_dimension_x[i],bin_dimension_y[i]};
            
  
  int bdx=bin_dimension_x[i];
  //    loop through grid
  for(int yi = corner.y;yi<corner.y+bin_dimension_x[i];yi+=1){
    int y=(yi-corner.y+jj)%bdx+corner.y; //shift so that there is no overlap
    
    //	  int y=yi;
    for(int x = corner.x;x<corner.x+bin_dimension_y[i];x+=1){
      
      sum_t[tid] = 0;
      
      for(int j = tid+jj*bd ;j<ppb;j+=bd*gridDim.x){
	sum_t[tid] += point_value[binned_points[idx+j]]*
	  kb_weight(make_float2(x,y),
		    make_float2(binned_points_x[idx+j],binned_points_y[idx+j]),
		    kb_table_size,kb_table_scale, kb_table,texRef);
      }
      __syncthreads();
      
      for(unsigned int j=1; j < bd; j *= 2) {
	// modulo arithmetic is slow!
	if ((tid & (2*j-1)) == 0) { sum_t[tid] += sum_t[tid + j];  }
	__syncthreads();
      }
      
      cudaDeviceSynchronize();
      
      if(tid == 0){
	//	  grid_value[y*grid_size.x+x]+=(cusp::complex<float>) sum_t[0]; 
	atomicAdd(&(grid_value[y*grid_size.x+x]),(sum_t[0]));
      }
    }
  }
}



//------------------------------
__global__ void grid_points_cuda_mex_interleaved_kernel(
        const cusp::complex<float> * point_value,
        int npoints,  uint2 grid_size,
        const int *  points_per_bin,
        const int * bin_dimension_x,
        const int * bin_dimension_y,
        const int *  binned_points,
        const int * binned_points_idx,
        const int * bin_location,
        const float * binned_points_x,
        const float * binned_points_y,
        const int nbins,
        const int kb_table_size,
							const float kb_table_scale,                 const float * kb_table,//cudaTextureObject_t texRef,
	size_t offset,size_t kernel_tex,
	cusp::complex<float> * grid_value){
  __shared__ cusp::complex<float> value;
      
    // Specialize BlockReduce for a 1D block of 128 threads on type cusp::complex<float>
    
    int i = blockIdx.x;
        if(points_per_bin[i]==0){return;}

    int tid = threadIdx.x;
    uint2 corner;
    corner.x = bin_location[i]%grid_size.x;
    corner.y = bin_location[i]/grid_size.x;

    const int idx = binned_points_idx[i];
    const int ppb = points_per_bin[i];
    //    cusp::complex<float> * value;
    const int  bd=BLOCKSIZE;
    //	const int  bd=blockDim.x;
    const uint2 dims = {bin_dimension_x[i],bin_dimension_y[i]};
            


    // small bin or large no of samples
    if(bin_dimension_x[i]*bin_dimension_y[i] < 64 || points_per_bin[i] > SHARED_SIZE){
      //      return;


	  /*
	  sum_points<<<3,BLOCKSIZE>>>( point_x, point_y,
                    point_value, npoints, grid_size, points_per_bin,
                    bin_dimension_x, bin_dimension_y, binned_points,
                    binned_points_idx, bin_location,
                    binned_points_x, binned_points_y,nbins,
                    kb_table_size,
		    kb_table_scale, kb_table, 
		  grid_value, blockIdx.x);
	  */
	  
          __shared__ cusp::complex<float> sum_t[bd];

//    loop through grid
        for(int y = corner.y;y<corner.y+bin_dimension_x[i];y+=1){
            for(int x = corner.x;x<corner.x+bin_dimension_y[i];x+=1){
        sum_t[tid] = 0;

        for(int j = tid;j<ppb;j+=bd){
	    sum_t[tid] +=   fetch_x(binned_points[idx+j],point_value)* 
		    //sum_t[tid] += point_value[binned_points[idx+j]]*
          kb_weight(make_float2(x,y),
                    make_float2(binned_points_x[idx+j],binned_points_y[idx+j]),
			      kb_table_size,kb_table_scale, kb_table);

        }
	//          __syncthreads();

        for(unsigned int j=1; j < bd; j *= 2) {
            // modulo arithmetic is slow!
            if ((tid & (2*j-1)) == 0) { sum_t[tid] += sum_t[tid + j];  }
	                __syncthreads();
        }

        if(tid == 0){
	  grid_value[y*grid_size.x+x]=sum_t[0]; 
	  //	atomicAdd(&(grid_value[y*grid_size.x+x]).x,(sum_t[0]).x);
	  //atomicAdd(&(grid_value[y*grid_size.x+x]).y,(sum_t[0]).y);

}
                
            }
        }        // large dimensions
    }else{ //small dimension and few points


  int bblock;
    if(bin_dimension_x[i]*bin_dimension_y[i] >BLOCKSIZE/2-1)
      bblock=1;
    else {bblock=4; };

    __shared__ float point_pos_cache_x[SHARED_SIZE];
    __shared__ float point_pos_cache_y[SHARED_SIZE];
    __shared__ cusp::complex<float> point_value_cache[SHARED_SIZE];  
    __shared__ cusp::complex<float> sum_t[BLOCKSIZE];
    //  

    //    bblock=4;

        // Lets try to load things to shared memory /
        for(int j = tid;j<ppb;j+= blockDim.x){
	  // const int point = binned_points[idx+j];
	      point_value_cache[j] =fetch_x(binned_points[idx+j],point_value); 
	      //              point_value_cache[j] = point_value[point];
            point_pos_cache_x[j] = binned_points_x[idx+j];
            point_pos_cache_y[j] = binned_points_y[idx+j];
        }
        __syncthreads();
        int b = bblock;// each thread takes care of 

        for(int k = tid/b;k<dims.x*dims.y;k+=blockDim.x/b){
            const int x = (k%(dims.x))+corner.x;
            const int y = (k/dims.x)+corner.y;
            sum_t[tid] = 0;
            //sum_i[tid] = 0;
            for(int j = (tid&(b-1));j<ppb;j+=b){
	      //                float w= kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale, kb_table,texRef);

                float w=kb_weight(make_float2(x,y),make_float2(point_pos_cache_x[j],point_pos_cache_y[j]),
			       kb_table_size,kb_table_scale, kb_table);


                sum_t[tid] += point_value_cache[j]*w;
            }
            // Do a reduce in shared memory 
                __syncthreads();

            for(unsigned int j=1; j < b; j = (j << 1)) {
                // modulo arithmetic is slow!
                if ((tid & ((j<<1)-1)) == 0) {
                    sum_t[tid] += sum_t[tid + j];
                }
            }
            if((tid&(b-1)) == 0){
                grid_value[y*grid_size.x+x] = sum_t[tid];
                
            }
        }
    }
}

//--------------------------------
        void grid_points_cuda_interleaved_mex(const cusp::complex<float> * point_value, int npoints,
                uint2 grid_size, const int * points_per_bin, const int * bin_dimension_x,
                const int * bin_dimension_y,
                const int * binned_points, const int * binned_points_idx, const int * bin_location,
                const float * binned_points_x, const float * binned_points_y,
                int nbins,
                const float * kb_table,
                const int kb_table_size, const float kb_table_scale, cusp::complex<float> * grid_value){
            cudaMemset(grid_value,0,sizeof(float2)*grid_size.x*grid_size.y);


           int grid = nbins;
            int block_size = BLOCKSIZE;
            clock_t t_i = clock();


  /*

  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  if(0){
 cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
 cudaArray * cuArray;
 cudaMallocArray(&cuArray, &channelDesc, kb_table_size, 1);

 cudaMemcpyToArray ( cuArray,  0,0,kb_table,kb_table_size*sizeof(float) , cudaMemcpyDeviceToDevice);

      resDesc.resType = cudaResourceTypeArray;
     resDesc.res.array.array = cuArray;

  }
  else{
   resDesc.resType = cudaResourceTypeLinear;
   resDesc.res.linear.devPtr = (void *)kb_table;

    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
 resDesc.res.linear.desc.x = 32; // bits per channel
 resDesc.res.linear.sizeInBytes =  kb_table_size*sizeof(float);
  }


  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = 1;

  // create texture object: we only have to do this once!
  cudaTextureObject_t texRef=0;
  cudaCreateTextureObject(&texRef, &resDesc, &texDesc, NULL);

//--------------------------------------------
//  if(1){
  // create texture object
 cudaResourceDesc res1Desc;
 memset(&res1Desc, 0, sizeof(res1Desc));
   res1Desc.resType = cudaResourceTypeLinear;
   res1Desc.res.linear.devPtr = (void *)point_value;
 res1Desc.res.linear.desc.f = cudaChannelFormatKindFloat;
 res1Desc.res.linear.desc.x = 32; // bits per channel
 res1Desc.res.linear.sizeInBytes =  npoints   *sizeof(cusp::complex<float>);

  cudaTextureDesc tex1Desc;
  memset(&tex1Desc, 0, sizeof(tex1Desc));
  tex1Desc.addressMode[0] = cudaAddressModeWrap;
  tex1Desc.addressMode[1] = cudaAddressModeWrap;
  tex1Desc.readMode = cudaReadModeElementType;
  tex1Desc.filterMode = cudaFilterModeLinear;
  tex1Desc.normalizedCoords = 0;

  // create texture object: we only have to do this once!
  cudaTextureObject_t tex_value=0;
  cudaCreateTextureObject(&tex_value, &res1Desc, &tex1Desc, NULL);
  //  }else{

  //  size_t = size_t(-1);
  //  cusp::complex<float> offset = 	cusp::complex<float>(-1);
  //  CUDA_SAFE_CALL(cudaBindTexture(&offset,tex_x_int,binned_points));





  //  }
//--------------------------------------------
  */

 //textures
size_t value_tex =  size_t(-1);
cudaBindTexture(&value_tex,tex_x_float,point_value);

size_t kernel_tex =  size_t(-1);
cudaBindTexture(&kernel_tex,tex_x_float1,kb_table);


            grid_points_cuda_mex_interleaved_kernel<<<grid,block_size>>>( point_value, npoints, grid_size, points_per_bin,
                    bin_dimension_x, bin_dimension_y, binned_points,
                    binned_points_idx, bin_location,
                    binned_points_x, binned_points_y,nbins,
                    kb_table_size,
		    kb_table_scale,kb_table,value_tex,kernel_tex,
  		  grid_value);
            cudaThreadSynchronize();
            
	    // cudaDestroyTextureObject(texRef);
 cudaDestroyTextureObject(kernel_tex);
 cudaDestroyTextureObject(value_tex);


            clock_t t_e = clock();
            error_handle();

        }
        

//#define	SX	    prhs[0]
//#define	SY	prhs[1]
#define	SV	prhs[0]
#define	GRID_DIM	prhs[1]
#define	SPB	prhs[2]
#define	BIN_DIM_X      prhs[3]
#define	BIN_DIM_Y     prhs[4]
#define	SIB     prhs[5]
#define	BSO     prhs[6]
#define	BL     prhs[7]
#define	BPX     prhs[8]
#define	BPY     prhs[9]
#define	KLUT     prhs[10]
#define	KLUTS     prhs[11]


void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){

      /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    
    // mxGPUArray const *samples_x;
    // mxGPUArray const *samples_y;
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

mwSize *grid_dim=(mwSize *)grid_dim0; 

//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim[0]),(grid_dim[1]));
//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim1[0]),(grid_dim1[1]));


// OUTPUT
mxGPUArray  *grid_values;



//samples_x = mxGPUCreateFromMxArray(SX);
//samples_y = mxGPUCreateFromMxArray(SY);
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
 int npoints =  (int)(mxGPUGetNumberOfElements(bin_points_x));
 int kernel_lookup_table_size = ( int)(mxGPUGetNumberOfElements(kernel_lookup_table));

mwSize ndim= 2;
  
 
// output:
//  float2 * grid_values;
//  float2 * gold_grid_values;
 
//  plhs[0] = jkt_new( grid_dim[0], grid_dim[1], mxSINGLE_CLASS, mxREAL,);

//grid_values= mxGPUCreateGPUArray(ndim,grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
grid_values= mxGPUCreateGPUArray(ndim, grid_dim,mxSINGLE_CLASS,mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

// now get the pointer or whatever it is
//const float *d_samples_x = (const float  *)(mxGPUGetDataReadOnly(samples_x));
//const  float *d_samples_y = (const float  *)(mxGPUGetDataReadOnly(samples_y));
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
float * d_kernel_lookup_table = ( float  *)(mxGPUGetDataReadOnly(kernel_lookup_table));
const uint2 grid_size = {grid_dim[0],grid_dim[1]};

//float2 * d_grid_values = (float2  *)(mxGPUGetData(grid_values));
cusp::complex<float> * d_grid_values = (cusp::complex<float>  *)(mxGPUGetData(grid_values));




grid_points_cuda_interleaved_mex(d_samples_values, npoints, 
				    grid_size, d_samples_per_bin, d_bin_dimensions_x, d_bin_dimensions_y,
				    d_samples_in_bin, d_bin_start_offset, d_bin_location, 
				    d_bin_points_x, d_bin_points_y,
				    nbins, d_kernel_lookup_table,
				    kernel_lookup_table_size,
				    kernel_lookup_table_scale,  d_grid_values);
//mexErrMsgTxt("gpuArray 2");



plhs[0] = mxGPUCreateMxArrayOnGPU(grid_values);

/*
 cudaDestroyTextureObject(texRef);
*/


// mxGPUDestroyGPUArray( samples_x);
// mxGPUDestroyGPUArray( samples_y);
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
