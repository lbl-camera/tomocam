#include <stdlib.h>
#include <stdio.h>

#include "polarsample.h"

texture<float, 1, cudaReadModeElementType> texRef;

__device__ float kb_weight1(float grid_pos, float point_pos,
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

__global__ void polarsample_kernel(complex_t * point_pos,
				   complex_t * grid_value, 
				   int npoints, uint2 grid_size,
				   int kb_table_size,
				   float kb_table_scale,
				   float kernel_radius,
				    complex_t * sample_value){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < npoints){

        // initialize sample values
        complex_t sv= make_cuFloatComplex(0.f, 0.f);

        // point positions
        float sx = point_pos[i].x;
        float sy = point_pos[i].y;

        // y-coord at the grid position in nbhd of current point
        int y = max(0, (int) ceil(sy - kernel_radius));
        int ymax = min((int) floor(sy + kernel_radius), (grid_size.y - 1));


        for ( ; y < ymax; y++ ) {
            if (y < 0 || y > grid_size.y-1) continue; 

            // calculate weight at y-coord
            float  kby = kb_weight1((float) y, sy, kb_table_size, kb_table_scale);
 
            // y-coord at the grid position in nbhd of current point
            int x = max(0, (int) ceil(sx - kernel_radius));
            int xmax = min((int) floor(sx + kernel_radius), (grid_size.x - 1));
            for(; x < xmax; x++) {
                float kbx = kb_weight1((float) x, sx, kb_table_size, kb_table_scale);
	            sv = sv + grid_value[y*grid_size.x+x]* kby * kbx;
            }
        }
        sample_value[i] = sv;
    }
}

void polarsample(complex_t * point_pos,
		 complex_t * grid_value, int npoints, 
		 uint2 grid_size,
		 float * kb_table,
		 int kb_table_size,
		 float kb_table_scale,
		 float kernel_radius, 
		 complex_t * sample_value){

  size_t offset;
  cudaMemset(sample_value,0,sizeof( complex_t)*npoints);
  cudaBindTexture(&offset,texRef, kb_table, sizeof(float)*kb_table_size);
  if(offset != 0){
    fprintf(stderr, "Table not allocated with cudaMalloc!\n");
    return;
  }

  int block_size = BLOCKSIZE;
  int grid = (npoints+block_size-1)/block_size;
  clock_t t_i = clock();
    polarsample_kernel<<<grid,block_size>>>(
            point_pos,
		    grid_value, npoints, 
		    grid_size,
		    kb_table_size,
		    kb_table_scale,
		    kernel_radius,
		    sample_value);
  cudaThreadSynchronize();
  clock_t t_e = clock();
  error_handle();
}
