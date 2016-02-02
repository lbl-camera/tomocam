#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "polargrid.h"
//#include "jacket.h"
#include "mex.h"



void grid_points_gold(float * d_point_pos_x, float * d_point_pos_y, float2 * d_point_value,
		      int npoints, uint2 grid_size, int * d_points_per_bin,int * d_binned_points, 
		      int * d_binned_points_idx, int * d_bin_location, int * d_bin_dim_x,
		      int * d_bin_dim_y,int nbins, float * d_kb_table,int kb_table_size,
		      float kb_table_scale,
		      float2 * d_grid_value){
  
  /* we're gonna receive almost all device pointers that we have to convert to CPU memory */

  float * point_pos_x = new float[npoints];
  cudaMemcpy(point_pos_x,d_point_pos_x,sizeof(float)*npoints,cudaMemcpyDeviceToHost);
  float * point_pos_y = new float[npoints];
  cudaMemcpy(point_pos_y,d_point_pos_y,sizeof(float)*npoints,cudaMemcpyDeviceToHost);

  float2 * point_value = new float2[npoints];
  cudaMemcpy(point_value,d_point_value,sizeof(float2)*npoints,cudaMemcpyDeviceToHost);
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

  float2 * grid_value = new float2[grid_size.x*grid_size.y];

  memset(grid_value,0,sizeof(float2)*grid_size.x*grid_size.y);
  float * kb_table = new float[kb_table_size];
  cudaMemcpy(kb_table,d_kb_table,sizeof(float)*kb_table_size,cudaMemcpyDeviceToHost);
    
  for(int i = 0;i<nbins;i++){
    uint2 corner;
    corner.x = bin_location[i]%grid_size.x;
    corner.y = bin_location[i]/grid_size.x;
    int idx = binned_points_idx[i];
    for(int y = corner.y;y<corner.y+bin_dim_y[i];y++){
      for(int x = corner.x;x<corner.x+bin_dim_x[i];x++){
	grid_value[y*grid_size.x+x].x = 0;
	grid_value[y*grid_size.x+x].y = 0;
	for(int j = 0;j<points_per_bin[i];j++){
	  grid_value[y*grid_size.x+x].x += point_value[binned_points[idx+j]].x*
	    cpu_kb_weight(make_float2(x,y),
			  make_float2(point_pos_x[binned_points[idx+j]],
				      point_pos_y[binned_points[idx+j]]),
			  kb_table,
			  kb_table_size,
			  kb_table_scale);
	  grid_value[y*grid_size.x+x].y += point_value[binned_points[idx+j]].y*
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

  cudaMemcpy(d_grid_value,grid_value,sizeof(float2)*grid_size.x*grid_size.y,cudaMemcpyHostToDevice);

}

err_t jktFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]){	


  if(nrhs < 14){
    mexPrintf("Insufficient number of arguments.\n");
    return errNone;
  }

  int argc = 0;
  float * samples_x;
  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxSINGLE_CLASS){
    mexPrintf("Error: Sample x coordinates must be a gsingle.\n");
    return errNone;
  }
  TRY( jkt_mem((void **)&samples_x, prhs[argc] ) );


  int npoints = jkt_numel(prhs[argc]);
  printf("npoints %d\n",npoints);
  
  argc++;
  if(jkt_matlab(prhs[argc]) == true || jkt_class(prhs[argc]) != mxSINGLE_CLASS){
    mexPrintf("Error: Sample y coordinates must be a gsingle.\n");
    return errNone;
  }
  float * samples_y;

  TRY( jkt_mem((void **)&samples_y, prhs[argc] ) );
  argc++;

  if(jkt_matlab(prhs[argc]) == true || jkt_complex(prhs[argc]) == false ||
     jkt_class(prhs[argc]) != mxSINGLE_CLASS){
    mexPrintf("Error: Sample values must must be a complex gsingle array.\n");
    return errNone;
  }
  float2 * samples_values;
  TRY( jkt_mem((void **)&samples_values, prhs[argc] ) );
  argc++;

  if(jkt_numel(prhs[argc]) != 2 ||  jkt_matlab(prhs[argc]) == false){
    mexPrintf("Error: Grid not 2x1 matlab matrix.\n");
    return errNone;
  }

  /* We'll assume 2D grids for the moment */
  double * grid_dim = mxGetPr(prhs[argc]);
  mexPrintf("Grid Dimensions %d x %d\n",lrint(grid_dim[0]),lrint(grid_dim[1]));
  argc++;
  
  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxINT32_CLASS){
    mexPrintf("Error: Samples per bin must be a gint32.\n");
    return errNone;
  }

  int * samples_per_bin;
  TRY( jkt_mem((void **)&samples_per_bin, prhs[argc] ) );
  argc++;

  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxINT32_CLASS){
    mexPrintf("Error: Bin dimensions x part must be a gint32.\n");
    return errNone;
  }
  int * bin_dimensions_x;
  TRY( jkt_mem((void **)&bin_dimensions_x, prhs[argc] ) );
  argc++;

  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxINT32_CLASS){
    mexPrintf("Error: Bin dimensions y part must be a gint32.\n");
    return errNone;
  }
  int * bin_dimensions_y;
  TRY( jkt_mem((void **)&bin_dimensions_y, prhs[argc] ) );
  argc++;

  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxINT32_CLASS){
    mexPrintf("Error: Samples in bin must be a gint32.\n");
    return errNone;
  }
  int * samples_in_bin;
  TRY( jkt_mem((void **)&samples_in_bin, prhs[argc] ) );
  argc++;

  if(jkt_matlab(prhs[argc]) == true || jkt_class(prhs[argc]) != mxINT32_CLASS){
    mexPrintf("Error: Bin dimensions x part must be a gint32.\n");
    return errNone;
  }
  int * bin_start_offset;
  TRY( jkt_mem((void **)&bin_start_offset, prhs[argc]));
  argc++;

  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxINT32_CLASS){
    mexPrintf("Error: Bin location must be a gint32.\n");
    return errNone;
  }
  int * bin_location;
  TRY( jkt_mem((void **)&bin_location, prhs[argc]));
  argc++;

  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxSINGLE_CLASS){
    mexPrintf("Error: Bin points x must be a gsingle.\n");
    return errNone;
  }

  float * bin_points_x;
  TRY( jkt_mem((void **)&bin_points_x, prhs[argc]));
  argc++;

  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxSINGLE_CLASS){
    mexPrintf("Error: Bin points y must be a gsingle.\n");
    return errNone;
  }
  float * bin_points_y;
  TRY( jkt_mem((void **)&bin_points_y, prhs[argc]));
  argc++;

  if(jkt_matlab(prhs[argc]) == true|| jkt_class(prhs[argc]) != mxSINGLE_CLASS){
    mexPrintf("Error: Lookup table must be a gsingle.\n");
    return errNone;
  }
  float * kernel_lookup_table;
  TRY( jkt_mem((void **)&kernel_lookup_table, prhs[argc]));
  int kernel_lookup_table_size = jkt_numel(prhs[argc]);  
  argc++;

  
  if(jkt_scalar(prhs[argc]) == false || jkt_matlab(prhs[argc]) == false){
    mexPrintf("Error: Kernel lookup table scale should be a matlab scalar.\n");
    return errNone;    
  }
  float kernel_lookup_table_scale = mxGetScalar(prhs[argc]);
  argc++;

  int nbins = jkt_numel(prhs[5]);
  printf("npoints %d\n",npoints);
  uint2 grid_size = {grid_dim[0],grid_dim[1]};

  /* Output */
  float2 * grid_values;
  plhs[0] = jkt_new( grid_dim[0], grid_dim[1], mxSINGLE_CLASS, true );
  TRY( jkt_mem((void **)&grid_values,  plhs[0]) );


  grid_points_cuda_interleaved_mex( samples_x, samples_y,
				    samples_values, npoints, 
				    grid_size, samples_per_bin, bin_dimensions_x, bin_dimensions_y,
				    samples_in_bin, bin_start_offset, bin_location, 
				    bin_points_x, bin_points_y,
				    nbins, kernel_lookup_table,
				    kernel_lookup_table_size,
				    kernel_lookup_table_scale,
				    grid_values);
  if(nlhs == 2){
    float2 * gold_grid_values;
    plhs[1] = jkt_new( grid_dim[0], grid_dim[1], mxSINGLE_CLASS, true );
    TRY( jkt_mem((void **)&gold_grid_values,  plhs[1]) );    
    
    grid_points_gold(samples_x, samples_y, samples_values,
		     npoints, grid_size, samples_per_bin,samples_in_bin, 
		     bin_start_offset, bin_location, bin_dimensions_x,
		     bin_dimensions_y, nbins, kernel_lookup_table, kernel_lookup_table_size,
		     kernel_lookup_table_scale,
		     gold_grid_values);

  }
  return errNone;
}

