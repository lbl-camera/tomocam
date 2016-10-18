#include <stdlib.h>
#include <stdio.h>

#include "polarsample.h"

texture<float, 1, cudaReadModeElementType> texRef;

__inline__ __device__ void atomicAdd(complex_t & arr,  complex_t  val) {
  atomicAdd(&(arr.x), val.x);
  atomicAdd(&(arr.y), val.y);
}

__device__ float kb_weight(float grid_pos, float point_pos,
				    int kb_table_size, float kb_table_scale){
  float dist_x = fabs(grid_pos - point_pos)*kb_table_scale;
  int ix = (int) dist_x;
  float fx = dist_x - ix;
  if(ix+1 < kb_table_size){
    return (tex1Dfetch(texRef, ix)*(1.0f-fx) + tex1Dfetch(texRef,ix+1)*(fx));     
  }
  return 0.0f;
}

__global__ void polarsample_transpose_kernel(const complex_t * point_pos,
					     const complex_t * sample_value, 
					     int npoints, uint2 grid_size,
					     int kb_table_size,
					     float kb_table_scale,
					     float kernel_radius,
					     complex_t * grid_value){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i < npoints){
    complex_t sv=  sample_value[i] ;
    complex_t pp=point_pos[i];
    float sx=pp.x;
    float sy=pp.y;

    int y = max(0, (int) ceil(sy - kernel_radius));
    int ymax = min((int) floor(sy + kernel_radius), grid_size.y - 1);
    for( ; y < ymax; y++ ) {
      if(y < 0 || y > grid_size.y-1) continue; 
      complex_t svy = sv * kb_weight((float) y, sy, kb_table_size, kb_table_scale);
      int x = max(0, (int) ceil(sy - kernel_radius));
      int xmax = min((int) floor(sx + kernel_radius), grid_size.x - 1);
      for( ; x < xmax; x++ )
          svy = svy * kb_weight((float) x, sx, kb_table_size, kb_table_scale);
	    atomicAdd(grid_value[y*grid_size.x+x],  svy);
    }
  }
}

void polarsample_transpose(
        complex_t * point_pos, 
        complex_t * sample_value, 
        int npoints, uint2 grid_size, 
        float * kb_table, 
        int kb_table_size, 
        float kb_table_scale, 
        float kernel_radius,
        complex_t * grid_value){

  size_t offset;
  cudaBindTexture(&offset, texRef, kb_table, sizeof(float)*kb_table_size);
  if(offset != 0){
    printf("Error: Texture offset different than zero. Table not allocated with cudaMalloc!\n");
    return;
  }

  int block_size = BLOCKSIZE;
  int grid = (npoints+block_size-1)/block_size;
  clock_t t_i = clock();
  polarsample_transpose_kernel<<<grid,block_size>>>(
          point_pos,
		  sample_value, 
          npoints, grid_size,
		  kb_table_size,
		  kb_table_scale,
		  kernel_radius,
		  grid_value);
  clock_t t_e = clock();
  error_handle();
}
