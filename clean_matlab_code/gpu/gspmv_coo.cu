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
// #define	NCOL    prhs[3]
 #define	NROW    prhs[3]
// #define	NNZ    prhs[5]
#define	XV    prhs[4]


/* Output Arguments */
#define	Y	plhs[0]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
    
    mxGPUArray const *Aval;
    mxGPUArray const *Acol;
    mxGPUArray const *Arow;
    mxGPUArray const *x;
    mxGPUArray  *y;
    
//     int nnzs = lrint(mxGetScalar(NCOL));
     int nrow = lrint(mxGetScalar(NROW));
//       mexPrintf("nrows=%d", nrow);
//     int nptr=nrows+1;
//     int nnz  = lrint(mxGetScalar(NNZ));
//     
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /*get matlab variables*/
    Aval = mxGPUCreateFromMxArray(VAL);
    Acol = mxGPUCreateFromMxArray(COL);
    Arow = mxGPUCreateFromMxArray(ROW);
    x    = mxGPUCreateFromMxArray(XV);
    
    int nnz=mxGPUGetNumberOfElements(Acol);
//    int nrow=mxGPUGetNumberOfElements(Arow);
    int ncol =mxGPUGetNumberOfElements(x);

    
    mxComplexity isXVreal = mxGPUGetComplexity(x);
    mxComplexity isAreal = mxGPUGetComplexity(Aval);
    const mwSize ndim= 1;
    const mwSize dims[]={(mwSize) (nrow)};

    if (isAreal!=isXVreal)
    {
        mexErrMsgTxt("Aval and X must have the same complexity");
        return;
    }

//    if(mxGPUGetClassID(Aval) != mxSINGLE_CLASS||
 //  mxGPUGetClassID(x)!= mxSINGLE_CLASS||
 //  mxGPUGetClassID(Arow)!= mxINT32_CLASS||
 //  mxGPUGetClassID(Acol)!= mxINT32_CLASS){
  //   mexErrMsgTxt("usage: gspmv(single, int32, int32, single )");
  //   return;
  //  }
    
    
    // single or double
    if(((mxGPUGetClassID(Aval) != mxSINGLE_CLASS || mxGPUGetClassID(x)!= mxSINGLE_CLASS) &&
            ((mxGPUGetClassID(Aval) != mxDOUBLE_CLASS) || mxGPUGetClassID(x)!= mxDOUBLE_CLASS))||
            mxGPUGetClassID(Arow)!= mxINT32_CLASS||    mxGPUGetClassID(Acol)!= mxINT32_CLASS){
        mexErrMsgTxt("usage: gspmv(single/double, int32, int32, single/double )");
        return;
    };
    
    
    //create output vector
//     mexPrintf("\ncreating nrows=%d", dims[0]);
    y = mxGPUCreateGPUArray(ndim,dims,mxGPUGetClassID(x),isAreal, MX_GPU_DO_NOT_INITIALIZE);
     
    
    /* wrap indices from matlab */
    typedef const int  TI;  /* the type for index */
    TI *d_col =(TI  *)(mxGPUGetDataReadOnly(Acol));
    TI *d_row =(TI  *)(mxGPUGetDataReadOnly(Arow));
    // wrap with thrust::device_ptr
    thrust::device_ptr<TI>    wrap_d_col  (d_col);
    thrust::device_ptr<TI>    wrap_d_row  (d_row);
    // wrap with array1d_view 
    typedef typename cusp::array1d_view< thrust::device_ptr<TI> >   idx2Av;
    // wrap index arrays
    idx2Av colIndex (wrap_d_col , wrap_d_col + nnz);
    idx2Av rowIndex (wrap_d_row , wrap_d_row + nnz);
           
    if (isAreal!=mxREAL){
                 if (mxGPUGetClassID(Aval) != mxSINGLE_CLASS)
                 {
                         typedef const cusp::complex<double> TA;  /* the type for A */
        typedef const cusp::complex<double> TXV; /* the type for X */
        typedef cusp::complex<double> TYV; /* the type for Y */

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
        val2Av valIndex (wrap_d_val , wrap_d_val + nnz);
        x2Av xIndex   (wrap_d_x   , wrap_d_x   + ncol);
        y2Av yIndex(wrap_d_y, wrap_d_y+ nrow);
//        y2Av yIndex(wrap_d_y, wrap_d_y+ ncol);
        
        // combine info in CSR matrix
        typedef  cusp::coo_matrix_view<idx2Av,idx2Av,val2Av> DeviceView;
        
        DeviceView As(nrow, ncol, nnz, rowIndex, colIndex, valIndex);
                
        // multiply matrix
        cusp::multiply(As, xIndex, yIndex);
                 
                 }else{
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
        val2Av valIndex (wrap_d_val , wrap_d_val + nnz);
        x2Av xIndex   (wrap_d_x   , wrap_d_x   + ncol);
        y2Av yIndex(wrap_d_y, wrap_d_y+ nrow);
//        y2Av yIndex(wrap_d_y, wrap_d_y+ ncol);
        
        // combine info in CSR matrix
        typedef  cusp::coo_matrix_view<idx2Av,idx2Av,val2Av> DeviceView;
        
        DeviceView As(nrow, ncol, nnz, rowIndex, colIndex, valIndex);
                
        // multiply matrix
        cusp::multiply(As, xIndex, yIndex);
                 }
    }
     else{
                 if (mxGPUGetClassID(Aval) != mxSINGLE_CLASS)
                 {
                             typedef const double TA;  /* the type for A */
        typedef const double TXV; /* the type for X */
        typedef double TYV; /* the type for Y */
   
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
        val2Av valIndex (wrap_d_val , wrap_d_val + nnz);
        x2Av xIndex   (wrap_d_x   , wrap_d_x   + ncol);
        //y2Av yIndex(wrap_d_y, wrap_d_y+ ncol);        
        y2Av yIndex(wrap_d_y, wrap_d_y+ nrow);
        
        // combine info in CSR matrix
        typedef  cusp::csr_matrix_view<idx2Av,idx2Av,val2Av> DeviceView;
        
        DeviceView As(nrow, ncol, nnz, rowIndex, colIndex, valIndex);
                
        // multiply matrix
        cusp::multiply(As, xIndex, yIndex);

        }else{ //real single precision
         
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
        val2Av valIndex (wrap_d_val , wrap_d_val + nnz);
        x2Av xIndex   (wrap_d_x   , wrap_d_x   + ncol);
        //y2Av yIndex(wrap_d_y, wrap_d_y+ ncol);        
        y2Av yIndex(wrap_d_y, wrap_d_y+ nrow);
        
        // combine info in CSR matrix
        typedef  cusp::csr_matrix_view<idx2Av,idx2Av,val2Av> DeviceView;
        
        DeviceView As(nrow, ncol, nnz, rowIndex, colIndex, valIndex);
                
        // multiply matrix
        cusp::multiply(As, xIndex, yIndex);
        
    }
     }
    Y = mxGPUCreateMxArrayOnGPU(y);
    
    mxGPUDestroyGPUArray(Aval);
    mxGPUDestroyGPUArray(Arow);
    mxGPUDestroyGPUArray(Acol);
    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(y);

    return;
}

