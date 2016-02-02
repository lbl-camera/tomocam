#ifndef _CUDA_SAMPLE_H_
#define _CUDA_SAMPLE_H_ 1

void cuda_sample(float * point_pos_x, float * point_pos_y,
		  cusp::complex<float> * grid_value, int npoints, 
		 uint2 grid_size,
		 float * kb_table,
		 int kb_table_size,
		 float kb_table_scale,
		 float kernel_radius,		 
		  cusp::complex<float> * sample_value);

void sample_points_gold(float * d_point_pos_x, float * d_point_pos_y,  cusp::complex<float> * d_grid_value,
			int npoints, uint2 grid_size, float * d_kb_table,int kb_table_size,
			float kb_table_scale,float kernel_radius,
			 cusp::complex<float> * d_sample_value);




#endif
