#ifndef _POLARGRID_H_
#define _POLARGRID_H_ 1

//const int max_points_per_bin = 4096;
const int max_points_per_bin = 4096;
//const int cache_size = 1300;49152;
const int BLOCKSIZE = 256;
const int GRIDSIZE = 4096*4;
const int SHARED_SIZE = 256;//49152;
//const int SHARED_SIZE = 1024;//49152;
//const int SHARED_SIZE = 512;//49152;
//const int SUM_SIZE = 1024;//49152;
const int SUM_SIZE = 256;//49152;

__host__ __device__ float weight(int2 grid_pos, float2 point_pos);
__host__ __device__ float weight(float2 grid_pos, float2 point_pos);


// 
// void grid_points_cuda_mex(float * point_pos_x, float * point_pos_y,
// 			  cusp::complex<float>  * point, int npoints, 
// 			  uint2 grid_size, int * points_per_bin, int * bin_dimension_x,
// 			  int * bin_dimensions_y,
// 			  int * binned_points, int * binned_points_idx, int * bin_location, 
// 			  float * binned_points_x, float * binned_points_y,
// 			  int nbins,
// 			  float * kb_table,
// 			  int kb_table_size,
// 			  float kb_table_scale,
// 			  cusp::complex<float> * grid);

void grid_points_cuda_interleaved_mex(const float * point_pos_x, const float * point_pos_y,
				      const cusp::complex<float> * point_value,  int npoints, 
				      uint2 grid_size, const int * points_per_bin, const int * bin_dimension_x,
				      const int * bin_dimensions_y,
				      const int * binned_points, const int * binned_points_idx, const int * bin_location, 
				      const float * binned_points_x, const float * binned_points_y,
				      int nbins,
				      const float * kb_table,
				      int kb_table_size,
				      float kb_table_scale,
				      cusp::complex<float> * grid_value);

void sample_points_gold(float * d_point_pos_x, float * d_point_pos_y, cusp::complex<float> * d_grid_value,
			int npoints, uint2 grid_size, float * d_kb_table,int kb_table_size,
			float kb_table_scale,float kernel_radius,
			cusp::complex<float> * d_sample_value);

float cpu_kb_weight(float2 grid_pos, float2 point_pos,
		    float * kb_table,
		    int kb_table_size,
		    float kb_table_scale);


#endif
