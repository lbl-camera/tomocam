#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cusp/complex.h>
#include <cub/cub.cuh>
#include <thrust/reduce.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"
#include "polargrid.h"

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
        return (tex1Dfetch<float>(texRef,ix)*(1.0f-fx) + tex1Dfetch<float>(texRef,ix+1)*(fx)) *
                (tex1Dfetch<float>(texRef,iy)*(1.0f-fy) + tex1Dfetch<float>(texRef,iy+1)*(fy));
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
        return (tex1Dfetch<float>(texRef,ix)*(1.0f-fx) + tex1Dfetch<float>(texRef,ix+1)*(fx)) *
                (tex1Dfetch<float>(texRef,iy)*(1.0f-fy) + tex1Dfetch<float>(texRef,iy+1)*(fy));
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
        return (tex1Dfetch<float>(texRef,tid)*(1.0f-fx) + tex1Dfetch<float>(texRef,tid)*(fx)) *
                (tex1Dfetch<float>(texRef,tid)*(1.0f-fy) + tex1Dfetch<float>(texRef,tid)*(fy));
    }
    return 0.0f;
}


__global__ void sum_points(const cusp::complex<float> * point_value,
        const int *  binned_points,
        const float * binned_points_x,
        const float * binned_points_y,
        int points_per_bin, int idx,
        int x, int y, int kb_table_size, float kb_table_scale, cusp::complex<float> * grid_value)
{
    int tid = threadIdx.x;
    cusp::complex<float> sum_t[BLOCKSIZE];
   
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
        
//--------------------        
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

//                    return ret;
//                    return sum_t[0];

    }




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
                float kb_table_scale,
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
 sum_points<<<1,BLOCKSIZE>>> (point_value,binned_points,binned_points_x,binned_points_y,idx,ppb,x,y,kb_table_size,kb_table_scale,grid_value+y*grid_size.x+x);
             
            /*
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
                                    kb_table_size,kb_table_scale);
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
                        */
                         
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
                        float w=                      kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale);
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
                        float w= kb_weight(x,y,point_pos_cache_x[j],point_pos_cache_y[j],kb_table_size,kb_table_scale);
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
            grid_points_cuda_mex_interleaved_kernel1<<<grid,block_size>>>( point_pos_x, point_pos_y,
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
        
        void compare_to_gold(float * gridded, float * gold_gridded, uint2 grid_size){
            for(int i =0;i<grid_size.x*grid_size.y;i++){
                if(fabs(gridded[i]-gold_gridded[i])/gridded[i] > 1e-5 &&
                        fabs(gridded[i]-gold_gridded[i]) > 1e-7){
//      printf("cuda[%d] = %e gold[%d] = %e\n",i,gridded[i],i,gold_gridded[i]);
                    exit(1);
                }
            }
        }
        
        
        
