#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
//#include "cutil.h"
#include "cuda.h"
#include <cusp/complex.h>
#include <cusp/blas.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "polargrid.h"
#include <cuda_runtime.h>

//-- kernel
texture<float, 1, cudaReadModeElementType> texRef;

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

__host__ __device__ float weight(int2 grid_pos, float2 point_pos){
    return weight(make_float2(grid_pos.x,grid_pos.y),point_pos);
}

__host__ __device__ float weight(float2 grid_pos, float2 point_pos){
    if(fabs(grid_pos.x-point_pos.x) >= 3.0f ||
            fabs(grid_pos.y-point_pos.y) >= 3.0f){
        return 0;
    }
    return fabs(grid_pos.x-point_pos.x)+
            fabs(grid_pos.y-point_pos.y);
}

__device__ float kb_weight(float2 grid_pos, float2 point_pos,
        int kb_table_size,
        float kb_table_scale){
    float dist_x = fabsf(grid_pos.x-point_pos.x)*kb_table_scale;
    float dist_y = fabsf(grid_pos.y-point_pos.y)*kb_table_scale;
    
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

__device__ float kb_weight(float grid_x, float grid_y, float point_pos_x,
        float point_pos_y,
        int kb_table_size,
        float kb_table_scale){
    float dist_x = fabsf(grid_x-point_pos_x)*kb_table_scale;
    float dist_y = fabsf(grid_y-point_pos_y)*kb_table_scale;
    
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

__device__ float kb_weight(float2 grid_pos, float2 point_pos,
        int kb_table_size,
        float kb_table_scale,int tid){
    float dist_x = fabsf(grid_pos.x-point_pos.x)*kb_table_scale;
    float dist_y = fabsf(grid_pos.y-point_pos.y)*kb_table_scale;
    float ix = rintf(dist_x);
    float fx = dist_x-ix;
    float iy = rintf(dist_y);
    float fy = dist_y-iy;
    
    if(ix+1 < kb_table_size && iy+1 < kb_table_size){
        return (tex1Dfetch(texRef,tid)*(1.0f-fx) + tex1Dfetch(texRef,tid)*(fx)) *
                (tex1Dfetch(texRef,tid)*(1.0f-fy) + tex1Dfetch(texRef,tid)*(fy));
    }
    return 0.0f;
}

__global__ void sum_points(const cusp::complex<float> * point_value,
        const int *  binned_points,
        const float * binned_points_x,
        const float * binned_points_y,
        int points_per_bin,
        int idx,
        int x, int y, int kb_table_size, float kb_table_scale, cusp::complex<float> * grid_value)
{
    int tid = threadIdx.x;
    cusp::complex<float> sum_t[BLOCKSIZE];
    
    /**/
// sum all points
        sum_t[tid] = 0;
        for(int j = tid;j<points_per_bin;j+=blockDim.x){
            sum_t[tid] += point_value[binned_points[idx+j]]*kb_weight(make_float2(x,y),
                    make_float2(binned_points_x[idx+j],
                    binned_points_y[idx+j]),
                    kb_table_size,kb_table_scale);
        }
        for(unsigned int j=1; j < blockDim.x; j *= 2) {
            // modulo arithmetic is slow!
            if ((tid & (2*j-1)) == 0) {
                sum_t[tid] += sum_t[tid + j];
            }
            __syncthreads();
        }
        if(tid == 0){
            grid_value[0]=sum_t[0];
            return;
        }
    }


__global__ void grid_points_cuda_mex_interleaved_kernel(const float * point_x,
        const float * point_y,
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
        int nbins,
        int kb_table_size,
        float kb_table_scale,
        cusp::complex<float> * grid_value){
    
    
    // Specialize BlockReduce for a 1D block of 128 threads on type cusp::complex<float>
    //typedef cub::BlockReduce<cusp::complex<float>, 128> BlockReduce;
    // Allocate shared memory for BlockReduce
    //__shared__ typename BlockReduce::TempStorage temp_storage;
    //int aggregate = BlockReduce(temp_storage).Sum(thread_data);
    
    int i = blockIdx.x;
    int tid = threadIdx.x;
    uint2 corner;
    corner.x = bin_location[i]%grid_size.x;
    corner.y = bin_location[i]/grid_size.x;
    int idx = binned_points_idx[i];
        const int ppb = points_per_bin[i];
    __shared__ float point_pos_cache_x[SHARED_SIZE];
    __shared__ float point_pos_cache_y[SHARED_SIZE];
    __shared__ cusp::complex<float> point_value_cache[SHARED_SIZE];  
    __shared__ cusp::complex<float> sum_t[BLOCKSIZE];
    
    // small bin or large no of samples
    if(bin_dimension_x[i]*bin_dimension_y[i] < 64 || points_per_bin[i] > SHARED_SIZE){
//    loop through grid
        for(int y = corner.y;y<corner.y+bin_dimension_x[i];y+=1){
            for(int x = corner.x;x<corner.x+bin_dimension_y[i];x+=1){
	      cusp::complex<float> * value= &grid_value[y*grid_size.x+x];
	           	       sum_points<<<1,BLOCKSIZE>>> (point_value,binned_points,binned_points_x,binned_points_y,idx,ppb,x,y,kb_table_size,kb_table_scale, value);
            }
        }
        // large dimensions
    }else if(bin_dimension_x[i]*bin_dimension_y[i] >BLOCKSIZE/2-1) {
        // Lets try to load all points to shared memory /
 //       const int ppb = points_per_bin[i];
        for(int j = tid;j<ppb;j+= blockDim.x){
            const int point = binned_points[idx+j];
            point_value_cache[j] = point_value[point];
            point_pos_cache_x[j] = binned_points_x[idx+j];
            point_pos_cache_y[j] = binned_points_y[idx+j];
        }
        __syncthreads();
        const uint2 dims = {bin_dimension_x[i],bin_dimension_y[i]};
        // loop through dimensions
        for(int k = tid;k<dims.x*dims.y;k+=blockDim.x){
            const int x = (k%(dims.x))+corner.x;
            const int y = (k/dims.x)+corner.y;
            cusp::complex<float> my_sum = 0;
            for(int j = 0;j<ppb;j++){ //loop through all the points
                float w=                      kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale);
                my_sum += point_value_cache[j]*w;
            }
            grid_value[y*grid_size.x+x] = my_sum;
        }
    }else{ //small dimension and few points
        // Lets try to load things to shared memory /
        //const int ppb = points_per_bin[i];
        for(int j = tid;j<ppb;j+= blockDim.x){
            const int point = binned_points[idx+j];
            point_value_cache[j] = point_value[point];
            point_pos_cache_x[j] = binned_points_x[idx+j];
            point_pos_cache_y[j] = binned_points_y[idx+j];
        }
        __syncthreads();
        const uint2 dims = {bin_dimension_x[i],bin_dimension_y[i]};
        int b = 4;
        for(int k = tid/b;k<dims.x*dims.y;k+=blockDim.x/b){
            const int x = (k%(dims.x))+corner.x;
            const int y = (k/dims.x)+corner.y;
            sum_t[tid] = 0;
            //sum_i[tid] = 0;
            for(int j = (tid&(b-1));j<ppb;j+=b){
                float w= kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale);
                sum_t[tid] += point_value_cache[j]*w;
            }
            // Do a reduce in shared memory 
            for(unsigned int j=1; j < b; j = (j << 1)) {
                // modulo arithmetic is slow!
                if ((tid & ((j<<1)-1)) == 0) {
                    sum_t[tid] += sum_t[tid + j];
                    
                }
                __syncthreads();
            }
            if((tid&(b-1)) == 0){
                grid_value[y*grid_size.x+x] = sum_t[tid];
                
            }
        }
    }
}

__global__ void grid_points_cuda_mex_interleaved_kernel1(const float * point_x,
        const float * point_y,
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
        int nbins,
        int kb_table_size,
        float kb_table_scale,
        cusp::complex<float> * grid_value){
    __shared__ float point_pos_cache_x[SHARED_SIZE];
    __shared__ float point_pos_cache_y[SHARED_SIZE];
    __shared__ cusp::complex<float> point_value_cache[SHARED_SIZE];  
    __shared__ cusp::complex<float> sum_t[BLOCKSIZE];

    
    // Specialize BlockReduce for a 1D block of 128 threads on type cusp::complex<float>
    //typedef cub::BlockReduce<cusp::complex<float>, 128> BlockReduce;
    // Allocate shared memory for BlockReduce
    //__shared__ typename BlockReduce::TempStorage temp_storage;
    //int aggregate = BlockReduce(temp_storage).Sum(thread_data);
    
    int i = blockIdx.x;
    int tid = threadIdx.x;
    uint2 corner;
    corner.x = bin_location[i]%grid_size.x;
    corner.y = bin_location[i]/grid_size.x;
    int idx = binned_points_idx[i];
    //    int ppb=points_per_bin[i];
     const int ppb = points_per_bin[i];


    // small bin or large no of samples
    if(bin_dimension_x[i]*bin_dimension_y[i] < 64 || ppb > SHARED_SIZE){
//    loop through grid
         for(int y = corner.y;y<corner.y+bin_dimension_x[i];y+=1){
            for(int x = corner.x;x<corner.x+bin_dimension_y[i];x+=1){

	      cusp::complex<float> * value= &grid_value[y*grid_size.x+x];

	          	              sum_points<<<1,BLOCKSIZE>>> (point_value,binned_points,binned_points_x,binned_points_y,idx,ppb,x,y,kb_table_size,kb_table_scale, value);
          
            }
        }

         // large dimensions
     }else if(bin_dimension_x[i]*bin_dimension_y[i] >BLOCKSIZE/2-1) {

         // Lets try to load all points to shared memory /
      //  return;
         for(int j = tid;j<ppb;j+= blockDim.x){
             const int point = binned_points[idx+j];
              point_value_cache[j] = point_value[point];
             point_pos_cache_x[j] = binned_points_x[idx+j];
             point_pos_cache_y[j] = binned_points_y[idx+j];
        }
        __syncthreads();
         const uint2 dims = {bin_dimension_x[i],bin_dimension_y[i]};
        // loop through dimensions
         for(int k = tid;k<dims.x*dims.y;k+=blockDim.x){
             const int x = (k%(dims.x))+corner.x;
             const int y = (k/dims.x)+corner.y;
             cusp::complex<float> my_sum = 0;
             for(int j = 0;j<ppb;j++){ //loop through all the points
                 float w=                      kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale);
                 my_sum += point_value_cache[j]*w;
             }
             grid_value[y*grid_size.x+x] = my_sum;

         }
    }else{

      //                  return;
 //small dimension and few points
         // Lets try to load things to shared memory /
        const int ppb = points_per_bin[i];
        for(int j = tid;j<ppb;j+= blockDim.x){
            const int point = binned_points[idx+j];
            point_value_cache[j] = point_value[point];
            point_pos_cache_x[j] = binned_points_x[idx+j];
            point_pos_cache_y[j] = binned_points_y[idx+j];
        }
        __syncthreads();
        const uint2 dims = {bin_dimension_x[i],bin_dimension_y[i]};
        int b = 4;
        for(int k = tid/b;k<dims.x*dims.y;k+=blockDim.x/b){
            const int x = (k%(dims.x))+corner.x;
            const int y = (k/dims.x)+corner.y;
            sum_t[tid] = 0;
            //sum_i[tid] = 0;
            for(int j = (tid&(b-1));j<ppb;j+=b){
                float w= kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale);
                sum_t[tid] += point_value_cache[j]*w;
            }
            // Do a reduce in shared memory 
            for(unsigned int j=1; j < b; j = (j << 1)) {
                // modulo arithmetic is slow!
                if ((tid & ((j<<1)-1)) == 0) {
                    sum_t[tid] += sum_t[tid + j];
                    
                }
                __syncthreads();
            }
            if((tid&(b-1)) == 0){
                grid_value[y*grid_size.x+x] = sum_t[tid];
                
            }
        }
    }
}
 
void grid_points_cuda_interleaved_mex(const float * point_pos_x, const float * point_pos_y,
                 const cusp::complex<float> * point_value, int npoints,
                 uint2 grid_size, const int * points_per_bin, const int * bin_dimension_x,
                  const int * bin_dimension_y,
                 const int * binned_points, const int * binned_points_idx, const int * bin_location,
                 const float * binned_points_x, const float * binned_points_y,
                 int nbins,
                 const float * kb_table,
                 int kb_table_size,
                 float kb_table_scale,
                 cusp::complex<float> * grid_value){
             cudaMemset(grid_value,0,sizeof(float2)*grid_size.x*grid_size.y);
            
             size_t offset;
             cudaBindTexture(&offset,texRef, kb_table, sizeof(float)*kb_table_size);
             if(offset != 0){
                  //   printf("Error: Texture offset different than zero. Table not allocated with cudaMalloc!%d\n");
                 return;
             }
            
           int grid = nbins;
           int block_size = BLOCKSIZE;
           clock_t t_i = clock();
           grid_points_cuda_mex_interleaved_kernel<<<grid,block_size>>>( point_pos_x, point_pos_y,
                   point_value, npoints, grid_size, points_per_bin,
                   bin_dimension_x, bin_dimension_y, binned_points,
                   binned_points_idx, bin_location,
                   binned_points_x, binned_points_y,nbins,
                   kb_table_size,
                   kb_table_scale,
                   grid_value);
           cudaThreadSynchronize();
            
           clock_t t_e = clock();
           error_handle();
           //  printf("%d iter in %5.1f ms\n",iter,(t_e-t_i)*1000.0/CLOCKS_PER_SEC);
       }
        
        

//--

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

mwSize *grid_dim=(mwSize *)grid_dim0; 

//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim[0]),(grid_dim[1]));
//mexPrintf("Grid Dimensions %d x %d\n",(grid_dim1[0]),(grid_dim1[1]));


// OUTPUT
mxGPUArray  *grid_values;
//mxGPUArray  *grid_values, *gold_grid_values;



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

mwSize ndim= 2;
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


