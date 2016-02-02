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
#define	PTR	prhs[2]
#define	XV    prhs[3]

/* Output Arguments */
#define	Y	plhs[0]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs,const mxArray * prhs[]){
    
    mxGPUArray const *Aval;
    mxGPUArray const *Acol;
    mxGPUArray const *Aptr;
    mxGPUArray const *x;
    mxGPUArray  *y;
    
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /*get matlab variables*/
    Aval = mxGPUCreateFromMxArray(VAL);
    Acol = mxGPUCreateFromMxArray(COL);
    Aptr = mxGPUCreateFromMxArray(PTR);
    x    = mxGPUCreateFromMxArray(XV);
    
    int ncol=mxGPUGetNumberOfElements(Acol);
    int nptr=mxGPUGetNumberOfElements(Aptr);
    int nin =mxGPUGetNumberOfElements(x);
    
    mxComplexity isXVreal = mxGPUGetComplexity(x);
    mxComplexity isAreal = mxGPUGetComplexity(Aval);
    
    
    
    typedef float TA;  /* the type for A */
    typedef float TXV; /* the type for X */
    typedef float TYV; /* the type for Y */
    
    const mwSize ndim= 1;
    const mwSize dims[]={nin};
    y = mxGPUCreateGPUArray(ndim,dims,mxGPUGetClassID(x),mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    
    /* pointers from matlab */
    const int                  *d_col =(const int  *)(mxGPUGetDataReadOnly(Acol));
    const int                  *d_ptr =(const int  *)(mxGPUGetDataReadOnly(Aptr));
    const TA *d_val =(const TA  *)(mxGPUGetDataReadOnly(Aval));
    const TXV *d_x   =(const TXV  *)(mxGPUGetDataReadOnly(x));
    TYV       *d_y   =(TYV  *)(mxGPUGetData(y));
    
    // wrap with thrust::device_ptr!
    thrust::device_ptr<const int>    wrap_d_col  (d_col);
    thrust::device_ptr<const int>    wrap_d_ptr  (d_ptr);
    thrust::device_ptr<const TA >    wrap_d_val  (d_val);
    thrust::device_ptr<const TXV >    wrap_d_x    (d_x);
    thrust::device_ptr<TYV >    wrap_d_y  (d_y);
    // use array1d_view to wrap the individual arrays
    typedef typename cusp::array1d_view< thrust::device_ptr<const int> >   deviceIndexArrayView;
    //typedef typename cusp::array1d_view< thrust::device_ptr<      TA > >   deviceValueArrayView;
    typedef typename cusp::array1d_view< thrust::device_ptr<const TA > >   deviceCValueArrayView;
    typedef typename cusp::array1d_view< thrust::device_ptr<const TXV > >   deviceXVValueArrayView;
    typedef typename cusp::array1d_view< thrust::device_ptr<      TYV > >   deviceYValueArrayView;
    
// wrap the individual arrays
    deviceIndexArrayView colIndex (wrap_d_col , wrap_d_col + ncol);
    deviceCValueArrayView valIndex (wrap_d_val , wrap_d_val + ncol);
    deviceIndexArrayView ptrIndex (wrap_d_ptr , wrap_d_ptr + nptr);
    deviceXVValueArrayView xIndex   (wrap_d_x   , wrap_d_x   + nin);
    deviceYValueArrayView yIndex(wrap_d_y, wrap_d_y+ nin);
    
    // combine info in CSR matrix
    typedef
            cusp::csr_matrix_view<deviceIndexArrayView,deviceIndexArrayView,
            deviceCValueArrayView> DeviceView;
    
    DeviceView As(nin, nin, ncol, ptrIndex, colIndex, valIndex);
    
    
    // multiply matrix
    cusp::multiply(As, xIndex, yIndex);


    Y = mxGPUCreateMxArrayOnGPU(y);
    
    mxGPUDestroyGPUArray(Aval);
    mxGPUDestroyGPUArray(Aptr);
    mxGPUDestroyGPUArray(Acol);
    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(y);

    
    return;
}

