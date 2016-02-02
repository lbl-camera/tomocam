#include <cuda.h>
#include <cusp/blas.h>
#include<cusp/csr_matrix.h>
#include<cusp/multiply.h>
#include <cusp/array1d.h>
#include <cusp/copy.h>
#include <thrust/device_ptr.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"



template <typename IndexType>
        struct empty_row_functor
{
    typedef bool result_type;
    
    template <typename Tuple>
            __host__ __device__
            bool operator()(const Tuple& t) const
    {
        const IndexType a = thrust::get<0>(t);
        const IndexType b = thrust::get<1>(t);
        
        return a != b;
    }
};

/* Input Arguments */
#define	ROWPTR	prhs[0]
#define	NPTR    prhs[1]
#define	NNZ    prhs[2]

/* Output Arguments */
#define	ROW_OUT	plhs[0]


void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
    mxGPUArray  *Arow;
    mxGPUArray const *rowptr;
    mxInitGPU();     /* Initialize the MathWorks GPU API. */
    int nptr = lrint(mxGetScalar(NPTR));
    int nnz  = lrint(mxGetScalar(NNZ));
    const mwSize ndim= 1;
    const mwSize dimrow[]={mwSize(nnz)};
//      const mwSize dimptr[]={mwSize(nptr)};
//      mexPrintf("nrows=%d,nnz=%d\n", dimptr[0],dimcol[0]);
    
    // input output array
    rowptr = mxGPUCreateFromMxArray(ROWPTR);
    Arow  = mxGPUCreateGPUArray(ndim,dimrow,mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    int *d_rowptr =(int  *)(mxGPUGetDataReadOnly(rowptr));
    int *d_Arow =(int  *)(mxGPUGetData(Arow));
    
    // wrap with thrust::device_ptr
    thrust::device_ptr<int>    wd_Arow  (d_Arow);
    thrust::device_ptr<int>    wd_rowptr  (d_rowptr);

    /*-----------------------------------------------------------*/
    // ptr to row
    /*-----------------------------------------------------------*/
    
    thrust::fill(wd_Arow,wd_Arow+nptr, int(0));
    thrust::scatter_if( thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(nptr-1),
            wd_rowptr,
            thrust::make_transform_iterator(
            thrust::make_zip_iterator( thrust::make_tuple(wd_rowptr,wd_rowptr+1 ) ),
            empty_row_functor<int>()),
            wd_Arow);
    thrust::inclusive_scan(wd_Arow,wd_Arow+nnz, wd_Arow, thrust::maximum<int>());
    /*-----------------------------------------------------------*/
    
//bring back to matlab
    ROW_OUT = mxGPUCreateMxArrayOnGPU(Arow);
    //clean up
    mxGPUDestroyGPUArray(Arow);
    mxGPUDestroyGPUArray(rowptr);
    
    return;
}

