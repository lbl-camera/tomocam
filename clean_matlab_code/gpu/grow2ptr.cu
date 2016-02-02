
#include <cuda.h>
#include <cusp/blas.h>
#include<cusp/csr_matrix.h>
#include<cusp/multiply.h>
#include <cusp/array1d.h>
#include <cusp/copy.h>
#include <thrust/device_ptr.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"


/* Input Arguments */
#define	ROW	prhs[0]
#define	NPTR    prhs[1]
#define	NNZ    prhs[2]

/* Output Arguments */
#define	ROW_OUT	plhs[0]




void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
    mxGPUArray const *Arow;
    mxGPUArray  *rowptr;
    mxInitGPU();     /* Initialize the MathWorks GPU API. */
   int nptr = lrint(mxGetScalar(NPTR));
    int nnz  = lrint(mxGetScalar(NNZ));
    const mwSize ndim= 1;    
//    const mwSize dimcol[]={mwSize(nnz)};    
     const mwSize dimptr[]={mwSize(nptr)};    
//      mexPrintf("nrows=%d,nnz=%d\n", dimptr[0],dimcol[0]);

    // input output array 
    Arow = mxGPUCreateFromMxArray(ROW);
      rowptr = mxGPUCreateGPUArray(ndim,dimptr,mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
//          mexPrintf("created matrix, nrows=%d,nnz=%d\n", nptr,nnz);
    // pointer from matlab
    int *d_Arow =(int  *)(mxGPUGetDataReadOnly(Arow));    
    int *d_rowptr =(int  *)(mxGPUGetData(rowptr));

    // wrap with thrust::device_ptr
    thrust::device_ptr<int>    wrap_d_Arow  (d_Arow);
    thrust::device_ptr<int>    wrap_d_rowptr  (d_rowptr);
    // convert to ptr 
      thrust::lower_bound(wrap_d_Arow,
                        wrap_d_Arow+nnz,
                        thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>(nptr),
                        wrap_d_rowptr);
    //bring back to matlab
    ROW_OUT = mxGPUCreateMxArrayOnGPU(rowptr);
    //clean up
    mxGPUDestroyGPUArray(Arow);
    mxGPUDestroyGPUArray(rowptr);

    return;
}

