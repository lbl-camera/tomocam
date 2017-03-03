/*
 * S. V. Venkatakrishnan, ORNL 2017
 */

#include <stdio.h>
#include "cublas_v2.h"
#include "debug.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuComplex.h>
#include <stdlib.h>

/* macro for index calculations */

//#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

/* matrix size and thread dimensions */

#define SIZE_ROW 2048
#define SIZE_COL 2048
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define BLOCK_K 16 // square block of K size

typedef cuFloatComplex complex_t;

__device__  float FILTER[3][3]={{1.0/12,1.0/6,1.0/12},{1.0/6, 0, 1.0/6},{1.0/12,1.0/6,1.0/12}};

__device__ void deriv_potFunc(float delta, float *ret_val)
{
  float MRF_C=.001;
  float MRF_P=1.2;
  float MRF_Q = 2;
  float MRF_SIGMA = 1;
  float temp1,temp2,temp3;
  temp1=pow(fabs(delta),MRF_Q - MRF_P)/(MRF_SIGMA);
  temp2=pow(fabs(delta),MRF_Q - 1);
  temp3 = MRF_C + temp1;
  if(delta < 0)
    *ret_val=((-1*temp2/(temp3*MRF_SIGMA))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else if(delta > 0)
  {
    *ret_val= ((temp2/(temp3*MRF_SIGMA))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  }
  else
    *ret_val=MRF_Q/(MRF_SIGMA*MRF_C);

}

__global__ void GPU_naive( const int num_row,const int num_col, complex_t const * const in_img, complex_t  *out_img)
{

/* determine my threads's row and col indices in the global matrix */
  const int myrow = blockDim.x * blockIdx.x + threadIdx.x;
  const int mycol = blockDim.y * blockIdx.y + threadIdx.y;
  float value=0;
  int k,l;  

/* if my row and col are not outside the boundary accumulate */
  if(myrow < num_row && mycol <num_col)
#pragma unroll    
   for (k=-1;k<2;k++)
     for (l=-1;l<2;l++)
       if(myrow+k>=0 && myrow+k<=num_row-1 && mycol+l >=0 && mycol+l <= num_col-1)
       {
	      deriv_potFunc((float)(in_img[INDX(myrow,mycol,num_col)].x-in_img[INDX(myrow+k,mycol+l,num_col)].x),&value);
              out_img[INDX(myrow,mycol,num_col)].x+=FILTER[k+1][l+1]*value;
	      deriv_potFunc((float)(in_img[INDX(myrow,mycol,num_col)].y-in_img[INDX(myrow+k,mycol+l,num_col)].y),&value);
              out_img[INDX(myrow,mycol,num_col)].y+=FILTER[k+1][l+1]*value;
       }          
  return;
} /* end GPU_naive */

void print_matrix(float *arr,int m,int n)
{
  int i,j;
  int count=0;
  for(i=0;i<m;i++)
    {
      for (j=0;j<n;j++)
	{
	  printf("%f ",arr[count++] );
	}
      printf("\n");
    }
}

int main( int argc, char *argv[] )
{

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  const int size_row = SIZE_ROW;
  const int size_col = SIZE_COL;

  fprintf(stdout, "Matrix size is %d\n",size_row*size_col);

  srand(time(NULL));

  complex_t *h_in_img, *h_out_img;
  complex_t *d_in_img, *d_out_img;
  complex_t sum = make_cuFloatComplex(0.f, 0.f);
  float elapsedTime;
 
  size_t numbytes = (size_t ) size_row * (size_t ) size_col * sizeof(complex_t);

  h_in_img = (complex_t *) malloc( numbytes );
  if( h_in_img == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_out_img = (complex_t *) malloc( numbytes );
  if( h_out_img == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }


/* initialize the A and B matrices */

  for( int i = 0; i < size_row * size_col; i++ )
  {
    h_in_img[i]  =  make_cuFloatComplex(1.0f, 1.0f);// / ( float(RAND_MAX) + 1.0 );
    h_out_img[i] =  make_cuFloatComplex(0, 0);
  }

/* allocate a, b, c in gpu memory */

  checkCUDA( cudaMalloc( (void **)&d_in_img, numbytes ) );
  checkCUDA( cudaMalloc( (void **)&d_out_img, numbytes ) );
  	
/* copy a and b to device */

  checkCUDA( cudaMemcpy( d_in_img, h_in_img, numbytes, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_out_img, h_out_img, numbytes, cudaMemcpyHostToDevice ) );

/* setup grid and block sizes */

  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( size_row / THREADS_PER_BLOCK_X + 1, 
               size_col / THREADS_PER_BLOCK_Y + 1, 1 );

/* start timers */
  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );

  printf("Starting kernel ...\n");
  checkCUDA( cudaEventRecord( start, 0 ) );
/* call GPU_naive */  
  GPU_naive<<< blocks, threads >>>(size_row,size_col,d_in_img, d_out_img);
  checkKERNEL()

/* stop timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  printf("Ending kernel compute ...\n");
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  printf("Time taken=%f s\n",elapsedTime/1000);

/* print data for GPU naive */

/* copy C back to host */
	
  checkCUDA( cudaMemcpy( h_out_img, d_out_img, numbytes, cudaMemcpyDeviceToHost ) );

  for (int i=0;i<size_row*size_col;i++)
      {	
//      sum+=h_out_img[i];
	sum.x+=h_out_img[i].x;
	sum.y+=h_out_img[i].y;
      }
  printf("Sum (real) = %f, Sum (imag) = %f\n",sum.x,sum.y);

  //  print_matrix(h_out_img,size_row,size_col);
  
  checkCUDA( cudaEventDestroy( start ) );
  checkCUDA( cudaEventDestroy( stop ) );


/* cleanup */

  checkCUDA( cudaFree( d_in_img ) );
  checkCUDA( cudaFree( d_out_img ) );
  free( h_in_img );
  free( h_out_img);
  checkCUDA( cudaDeviceReset() );
  return 0;
}
