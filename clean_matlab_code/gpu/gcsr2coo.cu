#include <cuda.h>
#include <cusp/complex.h>
#include <cusp/blas.h>
#include<cusp/csr_matrix.h>
#include<cusp/multiply.h>
#include <cusp/array1d.h>
#include <cusp/copy.h>
#include <thrust/device_ptr.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"


/* Input Arguments */
#define	VAL	prhs[0]
#define	COL	prhs[1]
#define	ROW	prhs[2]
#define	NCOL    prhs[3]
#define	NROW    prhs[4]
#define	NNZ    prhs[5]


/* Output Arguments */
#define	VAL_OUT	plhs[0]
#define	COL_OUT	plhs[1]
#define	ROW_OUT	plhs[2]


void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
    mxGPUArray const *Aval, *Acol, *Arow;
    mxGPUArray *Bval, *Bcol, *Brow;


    mxInitGPU();     /* Initialize the MathWorks GPU API. */
        
    
    int ncols = lrint(mxGetScalar(NCOL));
    int nrows = lrint(mxGetScalar(NROW));
    int nptr=nrows+1;
    int nnz  = lrint(mxGetScalar(NNZ));
    mexPrintf("nrows=%d,ncols=%d,nnz=%d\n", nrows,ncols,nnz);
    const mwSize ndim= 1;    
    const mwSize dimcol[]={mwSize(nnz)};    

    

    // input output array 
    Aval = mxGPUCreateFromMxArray(VAL);
    Acol = mxGPUCreateFromMxArray(COL);
    Arow = mxGPUCreateFromMxArray(ROW);
    int ncols1=mxGPUGetNumberOfElements(Acol);

   
    //create output array 
    Bcol = mxGPUCreateGPUArray(ndim,dimcol,mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    Bval = mxGPUCreateGPUArray(ndim,dimcol,mxGPUGetClassID(Aval), mxGPUGetComplexity(Aval), MX_GPU_DO_NOT_INITIALIZE);
    Brow = mxGPUCreateGPUArray(ndim,dimcol,mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);

        
    /* wrap indices from matlab */
    // wrap with array1d_view 
    typedef typename cusp::array1d_view< thrust::device_ptr<int> >   idx2Av;
       
    // from matlab
    int *d_Acol =(int  *)(mxGPUGetDataReadOnly(Acol));
    int *d_Arow =(int  *)(mxGPUGetDataReadOnly(Arow));    
    int *d_Bcol =(int  *)(mxGPUGetData(Bcol));
    int *d_Brow =(int  *)(mxGPUGetData(Brow));

    // wrap with thrust::device_ptr
    thrust::device_ptr<int>    wrap_d_Acol  (d_Acol);
    thrust::device_ptr<int>    wrap_d_Arow  (d_Arow);
    thrust::device_ptr<int>    wrap_d_Bcol  (d_Bcol);
    thrust::device_ptr<int>    wrap_d_Brow  (d_Brow);
    // wrap index arrays
    idx2Av AcolIndex (wrap_d_Acol , wrap_d_Acol + nnz);
    idx2Av ArowIndex (wrap_d_Arow , wrap_d_Arow + nptr); //this is CSR
    idx2Av Bcol_Index (wrap_d_Bcol , wrap_d_Bcol + nnz);
    //idx2Av Brow_Index (wrap_d_Brow , wrap_d_Brow + nptr); //for CSR
    idx2Av Brow_Index (wrap_d_Brow , wrap_d_Brow + nnz); //for COO

    
       
    //if (isAreal!=mxREAL){

        typedef  cusp::complex<float> TA;  /* the type for A */
        typedef typename cusp::array1d_view< thrust::device_ptr<TA > >   val2Av;
        typedef typename cusp::array1d_view< thrust::device_ptr<TA > >   val2Av1;

        /* pointers from matlab */
        TA *d_val =(TA  *)(mxGPUGetDataReadOnly(Aval));
        TA *d_Bval =(TA  *)(mxGPUGetData(Bval));
        // wrap with thrust::device_ptr
        thrust::device_ptr<TA >    wrap_d_val  (d_val);       
        thrust::device_ptr<TA >    wrap_d_Bval  (d_Bval);
        // wrap  with array1d
        val2Av valIndex (wrap_d_val , wrap_d_val + nnz);
        val2Av1 Bval_Index (wrap_d_Bval , wrap_d_Bval + nnz);
        
        // combine info in CSR or COO matrix
        typedef  cusp::csr_matrix_view<idx2Av,idx2Av,val2Av1> DeviceViewCSR;
        typedef  cusp::coo_matrix_view<idx2Av,idx2Av,val2Av> DeviceViewCOO;

        DeviceViewCSR As(nrows, ncols, nnz,  ArowIndex,   AcolIndex,   valIndex);
        DeviceViewCOO Bs(nrows, ncols, nnz, Brow_Index, Bcol_Index, Bval_Index);
        cusp::convert(As,Bs); 


cudaMemcpy(Acol,thrust::raw_pointer_cast(&CSR.column_indices[0]),sizeof(int)*(NNZa),cudaMemcpyDeviceToHost);
        
        
//         cudaMemcpy(Bcol1,thrust::raw_pointer_cast(d_Bcol),sizeof(int)*(nnz),cudaMemcpyDeviceToDevice);
//         cudaMemcpy(Brow1,thrust::raw_pointer_cast(d_Brow),sizeof(int)*(nptr),cudaMemcpyDeviceToDevice);
//         cudaMemcpy(Bval1,thrust::raw_pointer_cast(d_Bval),sizeof( cusp::complex<float>)*(nnz),cudaMemcpyDeviceToDevice);

    VAL_OUT = mxGPUCreateMxArrayOnGPU(Bval);
    COL_OUT = mxGPUCreateMxArrayOnGPU(Bcol);
    ROW_OUT = mxGPUCreateMxArrayOnGPU(Brow);


    
    mxGPUDestroyGPUArray(Aval);
    mxGPUDestroyGPUArray(Arow);
    mxGPUDestroyGPUArray(Acol);
    mxGPUDestroyGPUArray(Bval);
    mxGPUDestroyGPUArray(Brow);
    mxGPUDestroyGPUArray(Bcol);


    return;
}

