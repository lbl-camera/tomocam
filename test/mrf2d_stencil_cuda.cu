/*
 * S. V. Venkatakrishnan, ORNL 2017
 */

#include <stdio.h>
#include "cublas_v2.h"
#include "debug.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* macro for index calculations */

//#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

/* matrix size and thread dimensions */

#define SIZE_ROW 2048
#define SIZE_COL 2048
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define BLOCK_K 16 // square block of K size

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

/* Shared memory GPU kernel where each element of is computed by a single thread */

__global__ void GPU_shmem(const int num_row,const int num_col, float const * const in_img, float  *out_img)
{
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int iby = blockIdx.y * THREADS_PER_BLOCK_Y;
  const int ibx = blockIdx.x * THREADS_PER_BLOCK_X;

  /* shared memory arrays for A and B */
  __shared__ float in_s[(BLOCK_K + 2)*(BLOCK_K+2)];
  __shared__ float out_s[BLOCK_K*BLOCK_K];
  
/* determine my threads's row and col indices in the global matrix */

  const int myrow = blockDim.x * blockIdx.x + threadIdx.x;
  const int mycol = blockDim.y * blockIdx.y + threadIdx.y;
  float value=0;
  int k,l;

  int aoff = INDX( ibx + tx, ty, num_col );
  /* main loop over blocks of K */

  for( int Kblock = 0; Kblock < BLOCK_K; Kblock++ )
  {
/* read block of A into shared memory */
    in_s[ tx + blockDim.x*ty ] = in_img[ aoff ];
    __syncthreads();
    
  }

/* if my row and col are not in the boundary accumulate */

  if(myrow < num_row && mycol < num_col )
  {
      	for (k=-1;k<2;k++)
	  for (l=-1;l<2;l++)
	    {
	    deriv_potFunc(in_img[INDX(myrow,mycol,num_col)]-in_img[INDX(myrow+k,mycol+l,num_col)],&value);
	    out_img[INDX(myrow,mycol,num_col)]++;
	    //FILTER[k+1][l+1]*value;
	    }
    
  } /* end if */
  return;
} /* end GPU_naive */

__global__ void GPU_naive( const int num_row,const int num_col, float const * const in_img, float  *out_img)
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
	      deriv_potFunc(in_img[INDX(myrow,mycol,num_col)]-in_img[INDX(myrow+k,mycol+l,num_col)],&value);
	      //	      deriv_potFunc(in_img[INDX(myrow,mycol,num_row)]-in_img[INDX(myrow+k,mycol+l,num_row)],&value);
	    out_img[INDX(myrow,mycol,num_col)]++;//FILTER[k+1][l+1]*value;
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

  float *h_in_img, *h_out_img;
  float *d_in_img, *d_out_img;
  float sum=0;
  float elapsedTime;
 
  size_t numbytes = (size_t ) size_row * (size_t ) size_col * sizeof( float );

  h_in_img = (float *) malloc( numbytes );
  if( h_in_img == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_out_img = (float *) malloc( numbytes );
  if( h_out_img == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }


/* initialize the A and B matrices */

  for( int i = 0; i < size_row * size_col; i++ )
  {
    h_in_img[i] = 1;//fabs(rand());// / ( float(RAND_MAX) + 1.0 );
    h_out_img[i] = 0;
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
	sum+=h_out_img[i];
      }
  printf("Sum = %f\n",sum);

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
