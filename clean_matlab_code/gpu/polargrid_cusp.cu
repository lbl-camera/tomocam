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



/*
__inline__ __device__ void atomicAdd( cusp::complex<float> * x,  cusp::complex<float>  m)
{
     atomicAdd(&(x[0]).x,m.x);
     atomicAdd(&(x[0]).y,m.y);
}
*/

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

//---------
texture<float, 1, cudaReadModeElementType> texRef;
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

/*
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

//		     cudaDeviceSynchronize();

        if(tid == 0){
	  //	  grid_value[y*grid_size.x+x]+=(cusp::complex<float>) sum_t[0]; 
	  atomicAdd(&(grid_value[y*grid_size.x+x]),(sum_t[0]));

	}
	    }
	}
}
*/

//
// __device__ float kb_weight(float2 grid_pos, float2 point_pos,
//         float * kb_table, int kb_table_size,
//         float kb_table_scale){
//     float dist_x = fabsf(grid_pos.x-point_pos.x)*kb_table_scale;
//     float dist_y = fabsf(grid_pos.y-point_pos.y)*kb_table_scale;
//     int ix = (int)dist_x;
//     float fx = dist_x-rintf(dist_x);
//     int iy = (int)dist_y;
//     float fy = dist_y-rintf(dist_y);
//
//     if(ix+1 < kb_table_size && iy+1 < kb_table_size){
//         return (kb_table[ix]*(1.0f-fx) + kb_table[ix+1]*(fx)) *
//                 (kb_table[iy]*(1.0f-fy) + kb_table[iy+1]*(fy));
//     }
//     return 0.0f;
// }
/*
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
    __shared__ float point_pos_cache_x[SHARED_SIZE];
    __shared__ float point_pos_cache_y[SHARED_SIZE];
    __shared__ cusp::complex<float> point_value_cache[SHARED_SIZE];
   
    __shared__ cusp::complex<float> sum_t[BLOCKSIZE];
    
    // small bin or large no of samples
    if(bin_dimension_x[i]*bin_dimension_y[i] < 64 || points_per_bin[i] > SHARED_SIZE){
        sum_t[tid] = 0;
//    loop through grid
        for(int y = corner.y;y<corner.y+bin_dimension_x[i];y+=1){
            for(int x = corner.x;x<corner.x+bin_dimension_y[i];x+=1){
                sum_t[tid] = 0;
                // loop through points
                for(int j = tid;j<points_per_bin[i];j+=blockDim.x){
                    sum_t[tid] += point_value[binned_points[idx+j]]*kb_weight(make_float2(x,y),
                            make_float2(binned_points_x[idx+j],
                            binned_points_y[idx+j]),
                            kb_table_size,kb_table_scale);
                }
                // Do a reduce in shared memory //
                for(unsigned int j=1; j < blockDim.x; j *= 2) {
                    // modulo arithmetic is slow!
                    if ((tid & (2*j-1)) == 0) {
                        sum_t[tid] += sum_t[tid + j];
                    }
                    __syncthreads();
                }
                if(tid == 0){
                    grid_value[y*grid_size.x+x] = sum_t[0];
                }
            }
        }
        // large dimensions
    }else if(bin_dimension_x[i]*bin_dimension_y[i] >BLOCKSIZE/2-1) {
        // Lets try to load all points to shared memory /
        const int ppb = points_per_bin[i];
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
*/
//------------------------------
//point_value[binned_points[idx+j]]*kb_weight(make_float2(x,y),
//                           make_float2(binned_points_x[idx+j],
//                          binned_points_y[idx+j]),
//                          kb_table_size,kb_table_scale);

//call=
//transform_4in1out( binned_points_x,binned_points_y,binned_points,point_value,make_float2(x,y), kb_table_size,kb_table_scale);
 
/*
template <typename IN, typename OUT>
         struct KBMUL
        {
            float xs;
            float ys;
            int kb_table_size;
            float kb_table_scale;
                        
            KBMUL(float _xs, float _ys){
                xs = _xs;
                ys= _ys;
                kb_table_size=_kb_table_size;
                kb_table_scale=_kb_table_scale;
            }
                                
            template <typename Tuple>
                    __host__ __device__
                    OUT operator()(Tuple x)
            {
//                 OUT out;
                 IN point_value= thrust::get<0>(x);
                 float binned_points_x= thrust::get<1>(x);
                 float binned_points_y= thrust::get<2>(x);
                 
                   OUT  ret = point_value[binned_points]*kb_weight(make_float2(xs,ys),
                            make_float2(binned_points_x,binned_points_y),
                            kb_table_size,kb_table_scale);
                   return ret*ret;
            }
};
*/

//------------------------
/* 
 *                    sum_t[tid] += point_value[binned_points[idx+j]]*kb_weight(make_float2(x,y),
                            make_float2(binned_points_x[idx+j],
                            binned_points_y[idx+j]),
                            kb_table_size,kb_table_scale);

 * template <typename T1,typename T2>

 * struct AbsSubtract2 : public thrust::unary_function<T1,T2>
{
  __host__ __device__
  T2 operator()(T1 x)
  {  
    T2 ret = abs(abs(thrust::get<0>(x))-(thrust::get<1>(x)));
    return ret*ret;
  }
};
              */
//=========================
/*            
template<typename IN,typename OUT>
  void transform_4in_1out(float * BINNED_POINTS_X, float * BINNED_POINTS_Y, int BINNED_POINTS, IN
        * POINT_VALUE, float2 * POSITIONS, 
        int kb_table_size, float kb_table_scale, float * KBTABLE, OUT * derr, int N){
    thrust::device_ptr<float> d_binned_points_x(BINNED_POINTS_X);
    thrust::device_ptr<float> d_binned_points_y(BINNED_POINTS_Y);
    thrust::device_ptr<int> d_binned_points(BINNED_POINTS);
    thrust::device_ptr<float> d_value(POINT_VALUE);
    thrust::device_ptr<float> d_positions(POSITIONS);

    
}
//          transform_3in_2out(d_G,d_DG,d_a,  (float ) tau, &der,&d2er,n);
//          transform_3in_1out(d_G,d_DG,d_a,  (float ) tau, &der,n);

        template<typename IN,typename OUT>
//                void transform_3in_2out(IN * G, IN * dG, float * F, float tau, OUT * derr, OUT * d2err, int N){
                void transform_3in_2out(IN * G, IN * dG, float * F, float tau, OUT * derr, OUT * d2err, int N){
            thrust::device_ptr<IN> d_G(G);
            thrust::device_ptr<IN> d_dG(dG);
            thrust::device_ptr<float> d_F(F);
            thrust::tuple<OUT,OUT> init;
            thrust::tuple<OUT,OUT> out = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(d_G, d_dG, d_F)),
                    thrust::make_zip_iterator(thrust::make_tuple(d_G, d_dG, d_F))+N,
                    DIR<IN,OUT>(tau),
                    init,
                    TUPLE_PLUS<thrust::tuple<OUT,OUT> >());
            *derr = thrust::get<0>(out)*2;
            *d2err = thrust::get<1>(out)*2;
        }
*/




//--------------------------------
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
                float kb_table_scale,          const float * kb_table,
                cusp::complex<float> * grid_value){
            
            __shared__ float point_pos_cache_x[SHARED_SIZE];
            __shared__ float point_pos_cache_y[SHARED_SIZE];
            __shared__ cusp::complex<float> point_value_cache[SHARED_SIZE];
            
            __shared__ cusp::complex<float> sum_t[BLOCKSIZE];
            
            
            int i = blockIdx.x;
            int tid = threadIdx.x;
            uint2 corner;
            corner.x = bin_location[i]%grid_size.x;
            corner.y = bin_location[i]/grid_size.x;
            int idx = binned_points_idx[i];
           const int ppb = points_per_bin[i];
                         
            // small bin or large no of samples
            if(bin_dimension_x[i]*bin_dimension_y[i] < 64 || points_per_bin[i] > SHARED_SIZE){
//    loop through grid
                for(int y = corner.y;y<corner.y+bin_dimension_x[i];y+=1){
                    for(int x = corner.x;x<corner.x+bin_dimension_y[i];x+=1){
//                sum_points<<<1,BLOCKSIZE>>> (point_value,binned_points,binned_points_x,binned_points_y,idx,points_per_bin[idx],x,y,kb_table_size,kb_table_scale, value);
//             cusp::complex<float> value[1];
//grid_value[y*grid_size.x+x]=0;

// sum_points<<<1,BLOCKSIZE>>> (point_value,binned_points,binned_points_x,binned_points_y,idx,ppb,x,y,kb_table_size,kb_table_scale,grid_value+y*grid_size.x+x);
             
                sum_t[tid] = 0;
                        // Specialize BlockReduce for a 1D block of 128 threads on type cusp::complex<float>
                        //typedef cub::BlockReduce<cusp::complex<float>, 128> BlockReduce;
                        // Allocate shared memory for BlockReduce
                        //__shared__ typename BlockReduce::TempStorage temp_storage;
                        
                        //  grid_value[y*grid_size.x+x]= BlockReduce(temp_storage).Sum(thread_data);
                        //                for(int item=0; item<ITEMS_PER_THREAD; ++item)
//    data[item] = unaryOp(data[item]);
                        
                        sum_t[tid] = 0;
                        // loop through points
                        for(int j = tid;j<ppb;j+=blockDim.x){
                            sum_t[tid] += point_value[binned_points[idx+j]]*kb_weight(make_float2(x,y),
                                    make_float2(binned_points_x[idx+j],
                                    binned_points_y[idx+j]),
                                    kb_table_size,kb_table_scale,kb_table );
                        }
                        // Do a reduce in shared memory 
                        for(unsigned int j=1; j < blockDim.x; j *= 2) {
                            // modulo arithmetic is slow!
                            if ((tid & (2*j-1)) == 0) {
                                sum_t[tid] += sum_t[tid + j];
                            }
                            __syncthreads();
                        }
                        if(tid == 0){
                            grid_value[y*grid_size.x+x] = sum_t[0];
                        }
  
                         
                    }
                }
                // large dimensions
            }else if(bin_dimension_x[i]*bin_dimension_y[i] >BLOCKSIZE/2-1) {
                /* Lets try to load all points to shared memory */
                const int ppb = points_per_bin[i];
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
                        float w= kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale, kb_table);
                        my_sum += point_value_cache[j]*w;
                    }
                    grid_value[y*grid_size.x+x] = my_sum;
                }
            }else{ //small dimension and few points
                /* Lets try to load things to shared memory */
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
                        float w= kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale,kb_table);
                        sum_t[tid] += point_value_cache[j]*w;
                    }
                    /* Do a reduce in shared memory */
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
            /*
            grid_points_cuda_mex_interleaved_kernel1<<<grid,block_size>>>( point_pos_x, point_pos_y,
                    point_value, npoints, grid_size, points_per_bin,
                    bin_dimension_x, bin_dimension_y, binned_points,
                    binned_points_idx, bin_location,
                    binned_points_x, binned_points_y,nbins,
                    kb_table_size,
                    kb_table_scale, kb_table,  grid_value);
            cudaThreadSynchronize();
            */
            clock_t t_e = clock();
            error_handle();
            //  printf("%d iter in %5.1f ms\n",iter,(t_e-t_i)*1000.0/CLOCKS_PER_SEC);
        }

void compare_to_gold(float * gridded, float * gold_gridded, uint2 grid_size){
     for(int i =0;i<grid_size.x*grid_size.y;i++){
         if(fabs(gridded[i]-gold_gridded[i])/gridded[i] > 1e-5 &&
           fabs(gridded[i]-gold_gridded[i]) > 1e-7){
      printf("cuda[%d] = %e gold[%d] = %e\n",i,gridded[i],i,gold_gridded[i]);
                    exit(1);
                }
            }
        }

        
        
//---------




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

/*
grid_points_cuda_interleaved_mex( d_samples_x, d_samples_y,
				    d_samples_values, npoints, 
				    grid_size, d_samples_per_bin, d_bin_dimensions_x, d_bin_dimensions_y,
				    d_samples_in_bin, d_bin_start_offset, d_bin_location, 
				    d_bin_points_x, d_bin_points_y,
				    nbins, d_kernel_lookup_table,
				    kernel_lookup_table_size,
				    kernel_lookup_table_scale,
				    d_grid_values);
*/
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


