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
#define	A	prhs[0]
#define	XV    prhs[3]

/* Output Arguments */
#define	Y	plhs[0]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
     
    
    const char  *Avalm, *Acolm, *Aptrm;    
    const char *Anrow, *Ancol, *Annz;    
    
    mxGPUArray const *Aval;
    mxGPUArray const *Acol;
    mxGPUArray const *Aptr;

    mxGPUArray const *x;
    mxGPUArray  *y;
//     fnames[ifield] = mxGetFieldNameByNumber(A,ifield);
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /*get matlab pointers*/
    //Anrows= mxGetFieldNameByNumber(A,0);
    //Ancol=  mxGetFieldNameByNumber(A,1);
    //Annz=   mxGetFieldNameByNumber(A,2);
    Avalm= mxGetFieldNameByNumber(A,3);
    Acolm= mxGetFieldNameByNumber(A,4);
    Aptrm= mxGetFieldNameByNumber(A,5);
    
    /*get matlab variables*/
    Aval = mxGPUCreateFromMxArray(Avalm);
    Acol = mxGPUCreateFromMxArray(Acolm);
    Aptr = mxGPUCreateFromMxArray(Aptrm);
    x    = mxGPUCreateFromMxArray(XV);
     
    int ncol=mxGPUGetNumberOfElements(Acol);
    int nrowp1=mxGPUGetNumberOfElements(Aptr);
    int nnz =mxGPUGetNumberOfElements(x);

//    int nout=nrowp1-1;
    
    mxComplexity isXVreal = mxGPUGetComplexity(x);
    mxComplexity isAreal = mxGPUGetComplexity(Aval);
    const mwSize ndim= 1;
    const mwSize dims[]={(mwSize) (nrowp1-1)};
//    mxComplexity isYVreal=mxCOMPLEX;
//     if (isAreal==mxREAL && isXVreal==mxREAL) 
//         isYVreal=mxCOMPLEX;

            
    if (isAreal!=isXVreal)
    {
     mexErrMsgTxt("Aval and X must have the same complexity");
    return;
    }
    
    if(mxGPUGetClassID(Aval) != mxSINGLE_CLASS||
   mxGPUGetClassID(x)!= mxSINGLE_CLASS||
   mxGPUGetClassID(Aptr)!= mxINT32_CLASS||
   mxGPUGetClassID(Acol)!= mxINT32_CLASS){
     mexErrMsgTxt("usage: gspmv(single, int32, int32, single )");
     return;
    }
    
    //create output vector
    y = mxGPUCreateGPUArray(ndim,dims,mxGPUGetClassID(x),isAreal, MX_GPU_DO_NOT_INITIALIZE);
     
    
    /* wrap indices from matlab */
    typedef const int  TI;  /* the type for index */
    TI *d_col =(TI  *)(mxGPUGetDataReadOnly(Acol));
    TI *d_ptr =(TI  *)(mxGPUGetDataReadOnly(Aptr));
    // wrap with thrust::device_ptr
    thrust::device_ptr<TI>    wrap_d_col  (d_col);
    thrust::device_ptr<TI>    wrap_d_ptr  (d_ptr);
    // wrap with array1d_view 
    typedef typename cusp::array1d_view< thrust::device_ptr<TI> >   idx2Av;
    // wrap index arrays
    idx2Av colIndex (wrap_d_col , wrap_d_col + ncol);
    idx2Av ptrIndex (wrap_d_ptr , wrap_d_ptr + nrowp1);
           
    if (isAreal!=mxREAL){

        typedef const cusp::complex<float> TA;  /* the type for A */
        typedef const cusp::complex<float> TXV; /* the type for X */
        typedef cusp::complex<float> TYV; /* the type for Y */

        // wrap with array1d_view 
        typedef typename cusp::array1d_view< thrust::device_ptr<TA > >   val2Av;
        typedef typename cusp::array1d_view< thrust::device_ptr<TXV > >   x2Av;
        typedef typename cusp::array1d_view< thrust::device_ptr<TYV > >   y2Av;
        
        /* pointers from matlab */
        TA *d_val =(TA  *)(mxGPUGetDataReadOnly(Aval));
        TXV *d_x   =(TXV  *)(mxGPUGetDataReadOnly(x));
        TYV *d_y   =(TYV  *)(mxGPUGetData(y));
        
        // wrap with thrust::device_ptr
        thrust::device_ptr<TA >    wrap_d_val  (d_val);
        thrust::device_ptr<TXV >    wrap_d_x    (d_x);
        thrust::device_ptr<TYV >    wrap_d_y  (d_y);
        
        // wrap  arrays
        val2Av valIndex (wrap_d_val , wrap_d_val + ncol);
        x2Av xIndex   (wrap_d_x   , wrap_d_x   + nnz);
        y2Av yIndex(wrap_d_y, wrap_d_y+ nrowp1-1);
//        y2Av yIndex(wrap_d_y, wrap_d_y+ nnz);
        
        // combine info in CSR matrix
        typedef  cusp::csr_matrix_view<idx2Av,idx2Av,val2Av> DeviceView;
        
        DeviceView As(nrowp1-1, nnz, ncol, ptrIndex, colIndex, valIndex);
                
        // multiply matrix
        cusp::multiply(As, xIndex, yIndex);
        
    }
     else{
         
        typedef const float TA;  /* the type for A */
        typedef const float TXV; /* the type for X */
        typedef float TYV; /* the type for Y */
   
        /* pointers from matlab */
        TA *d_val =(TA  *)(mxGPUGetDataReadOnly(Aval));
        TXV *d_x   =(TXV  *)(mxGPUGetDataReadOnly(x));
        TYV *d_y   =(TYV  *)(mxGPUGetData(y));
        
        // wrap with thrust::device_ptr!
        thrust::device_ptr<TA >    wrap_d_val  (d_val);
        thrust::device_ptr<TXV >    wrap_d_x    (d_x);
        thrust::device_ptr<TYV >    wrap_d_y  (d_y);
        // wrap with array1d_view 
        typedef typename cusp::array1d_view< thrust::device_ptr<TA > >   val2Av;
        typedef typename cusp::array1d_view< thrust::device_ptr<TXV > >   x2Av;
        typedef typename cusp::array1d_view< thrust::device_ptr<TYV > >   y2Av;
        
        // wrap  arrays
        val2Av valIndex (wrap_d_val , wrap_d_val + ncol);
        x2Av xIndex   (wrap_d_x   , wrap_d_x   + nnz);
        //y2Av yIndex(wrap_d_y, wrap_d_y+ nnz);        
        y2Av yIndex(wrap_d_y, wrap_d_y+ nrowp1-1);
        
        // combine info in CSR matrix
        typedef  cusp::csr_matrix_view<idx2Av,idx2Av,val2Av> DeviceView;
        
        DeviceView As(nrowp1-1, nnz, ncol, ptrIndex, colIndex, valIndex);
                
        // multiply matrix
        cusp::multiply(As, xIndex, yIndex);
        
    }

    Y = mxGPUCreateMxArrayOnGPU(y);
    
    mxGPUDestroyGPUArray(Aval);
    mxGPUDestroyGPUArray(Aptr);
    mxGPUDestroyGPUArray(Acol);
    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(y);

    return;
}

