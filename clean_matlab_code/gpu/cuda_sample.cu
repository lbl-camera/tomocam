#include <cusp/complex.h>
#include "polargrid.h"
#include "cuda_sample.h"
#include "mex.h"
#include <stdio.h>

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


__global__ void cuda_sample_kernel(float * point_pos_x,
				   float * point_pos_y,
				   cusp::complex<float> * grid_value, 
				   int npoints, uint2 grid_size,
				   int kb_table_size,
				   float kb_table_scale,
				   float kernel_radius,
				    cusp::complex<float> * sample_value){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i < npoints){
    sample_value[i] = 0;
    for(int y = max(0.0f,ceil(point_pos_y[i]-kernel_radius));y<= min(floor(point_pos_y[i]+kernel_radius),grid_size.y-1.0f);y++){
      for(int x = max(0.0f,ceil(point_pos_x[i]-kernel_radius));x<= min(floor(point_pos_x[i]+kernel_radius),grid_size.x-1.0f);x++){
	if(x < 0 || x > grid_size.x-1){
	  continue;
	}
	if(y < 0 || y > grid_size.y-1){
	  continue;
	}
	sample_value[i] += grid_value[y*grid_size.x+x]*
	  kb_weight(make_float2(x,y),
		    make_float2(point_pos_x[i], point_pos_y[i]),
		    kb_table_size,
		    kb_table_scale);
	
      }
    }

  }
}

void cuda_sample(float * point_pos_x, float * point_pos_y,
		 cusp::complex<float> * grid_value, int npoints, 
		 uint2 grid_size,
		 float * kb_table,
		 int kb_table_size,
		 float kb_table_scale,
		 float kernel_radius,		 
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
  int iter = 1;
  for(int i = 0;i<iter;i++){
    cuda_sample_kernel<<<grid,block_size>>>( point_pos_x, point_pos_y,
					     grid_value, npoints, 
					     grid_size,
					     kb_table_size,
					     kb_table_scale,
					     kernel_radius,
					     sample_value);
    cudaThreadSynchronize();
    
  }
  clock_t t_e = clock();
  error_handle();
  //  printf("%d iter in %5.1f ms\n",iter,(t_e-t_i)*1000.0/CLOCKS_PER_SEC);
}

void sample_points_gold(float * d_point_pos_x, float * d_point_pos_y, cusp::complex<float> * d_grid_value,
			int npoints, uint2 grid_size, float * d_kb_table,int kb_table_size,
			float kb_table_scale,float kernel_radius,
			cusp::complex<float> * d_sample_value){
  
  /* we're gonna receive almost all device pointers that we have to convert to CPU memory */

  float * point_pos_x = new float[npoints];
  cudaMemcpy(point_pos_x,d_point_pos_x,sizeof(float)*npoints,cudaMemcpyDeviceToHost);
  float * point_pos_y = new float[npoints];
  cudaMemcpy(point_pos_y,d_point_pos_y,sizeof(float)*npoints,cudaMemcpyDeviceToHost);

  cusp::complex<float> * grid_value = new cusp::complex<float>[grid_size.x*grid_size.y];
  cudaMemcpy(grid_value,d_grid_value,sizeof(cusp::complex<float>)*grid_size.x*grid_size.y,cudaMemcpyDeviceToHost);

   cusp::complex<float> * sample_value = new cusp::complex<float>[npoints];

  memset(sample_value,0,sizeof(cusp::complex<float>)*npoints);
  float * kb_table = new float[kb_table_size];
  cudaMemcpy(kb_table,d_kb_table,sizeof(float)*kb_table_size,cudaMemcpyDeviceToHost);
    
  for(int i = 0;i<npoints;i++){
    for(int y = ceil(point_pos_y[i]-kernel_radius);y<= floor(point_pos_y[i]+kernel_radius);y++){
      for(int x = ceil(point_pos_x[i]-kernel_radius);x<= floor(point_pos_x[i]+kernel_radius);x++){
	if(x < 0 || x > grid_size.x-1){
	  continue;
	}
	if(y < 0 || y > grid_size.y-1){
	  continue;
	}

	sample_value[i] += grid_value[y*grid_size.x+x]*
	  cpu_kb_weight(make_float2(x,y),
			make_float2(point_pos_x[i], point_pos_y[i]),
			kb_table,
			kb_table_size,
			kb_table_scale);
	      }
    }
  }
  cudaMemcpy(d_sample_value,sample_value,sizeof(cusp::complex<float>)*npoints,cudaMemcpyHostToDevice);
}
