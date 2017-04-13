#include <cuda.h>
#include "polarsample.h"

const int DIMX = 16;
const int DIMY = 4;
const int DIMZ = 4;
const int WORK = 8;
const int sNX = DIMX + 2;
const int sNY = DIMY + 2;
const int sNZ = DIMZ + 2;
const float MRF_C = .001;
const float MRF_Q = 2;

__device__ const float FILTER[3][3][3] = 
    { 
        {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}},
        {{0.037,  0.0523, 0.037}, {0.0532, 0., 0.0523}, {0.037, 0.0523, 0.037}},
        {{0.0302, 0.037, 0.0302}, {0.037, 0.0523, 0.037}, {0.0302, 0.037, 0.0302}}
    };

__constant__ int nx, ny, nz;

__inline__ __device__ float wght(int k, int j, int i){
    return FILTER[k][j][i];
    
}

__inline__ __device__ int globalIdx(int x, int y, int z){
    return (nx * ny * z + nx * y + x);
}

__inline__ __device__ float pot_func(float delta, float MRF_P, float MRF_SIGMA)
{
  return ((pow(fabs(delta)/MRF_SIGMA,MRF_Q))/(MRF_C + pow(fabs(delta)/MRF_SIGMA,MRF_Q - MRF_P)));
}

__inline__ __device__ float deriv_potFCN(float delta, float MRF_P, float MRF_SIGMA) {
    float MRF_SIGMA_Q = pow(MRF_SIGMA,MRF_Q);
    float MRF_SIGMA_Q_P = pow(MRF_SIGMA,MRF_Q - MRF_P);

    float temp1 = pow(fabs(delta), MRF_Q - MRF_P) / MRF_SIGMA_Q_P;
    float temp2 = pow(fabs(delta), MRF_Q - 1);
    float temp3 = MRF_C + temp1;

    if(delta < 0.0) {
        return ((-1*temp2/(temp3*MRF_SIGMA_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
    } else if(delta > 0.0) {
        return ((temp2/(temp3*MRF_SIGMA_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
    } else {
      return 0; //MRF_Q / (MRF_SIGMA_Q*MRF_C);
    }
}

/*Second Derivative of the potential function at zero */
__inline__ __device__ float second_deriv_potFunc_zero(float MRF_SIGMA)
{
  float MRF_SIGMA_Q=pow(MRF_SIGMA,MRF_Q);
  return MRF_Q/(MRF_SIGMA_Q*MRF_C);
}

__global__ void tvd_update_kernel(float mrf_p, float mrf_sigma, complex_t * val, complex_t * tvd){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z; 
    int x = i + blockDim.x * blockIdx.x;
    int y = j + blockDim.y * blockIdx.y;
    int z = k + blockDim.z * blockIdx.z;

    // last thread in the block
    int in = min(nx - blockIdx.x*blockDim.x-1, blockDim.x-1);
    int jn = min(ny - blockIdx.y*blockDim.y-1, blockDim.y-1);
    int kn = min(nz - blockIdx.z*blockDim.z-1, blockDim.z-1);


    if ((x < nx) && (y < ny) && (z < nz)) {

        int gid = globalIdx(x, y, z);

        /* copy values into shared memory. 
         * Max size of shared memory = 64 x 1024 Bytes
         * which translates to  8192 complex number
         */

        const complex_t CMPLX_ZERO = make_cuFloatComplex(0.f, 0.f);
        __shared__ complex_t s_val[sNZ][sNY][sNX];

        // copy from global memory
        s_val[k+1][j+1][i+1] = val[gid];

        /* copy ghost cells, on all 6 faces */
        // x = 0 face
        if (i == 0){
            if (x > 0) s_val[k+1][j+1][i] = val[globalIdx(x-1, y, z)];
            else s_val[k+1][j+1][i] = CMPLX_ZERO;
        }

        // x = Nx-1 face
        if (i == in) {
            if (x < nx-1) s_val[k+1][j+1][i+2] = val[globalIdx(x+1, y, z)];
            else s_val[k+1][j+1][i+2] = CMPLX_ZERO;
        }
        __syncthreads();

        if (j == 0){
            if (y > 0) s_val[k+1][j][i+1] = val[globalIdx(x, y-1, z)];
            else s_val[k+1][j][i+1] = CMPLX_ZERO;
        }

        if (j == jn) {
            if (y < ny-1) s_val[k+1][j+2][i+1] = val[globalIdx(x, y+1, z)];
            else s_val[k+1][j+2][i+1] = CMPLX_ZERO;
        }
        __syncthreads();

        if (k == 0){
            if (z > 0) s_val[k][j+1][i+1] = val[globalIdx(x, y, z-1)];
            else s_val[k][j+1][i+1] = CMPLX_ZERO;
        }

        if (k == kn) {
            if (z < nz-1) s_val[k+2][j+1][i+1] = val[globalIdx(x, y, z+1)];
            else s_val[k+2][j+1][i+1] = CMPLX_ZERO;
        }
        __syncthreads();


        /* copy ghost cells along 12 edges  */
        // copy ghost-cells along x-direction
        if (j == 0) {
            if (k == 0) {
                if ((y > 0) && (z > 0))
                    s_val[k][j][i+1] = val[globalIdx(x, y-1, z-1)];
                else s_val[k][j][i+1] = CMPLX_ZERO;
            }
            if (k == kn) {
                if ((y > 0) && (z < nz-1))
                    s_val[k+2][j][i+1] = val[globalIdx(x, y-1, z+1)];
                else s_val[k+2][j][i+1] = CMPLX_ZERO;
            }
        }
        if (j == jn) {
            if (k == 0) {
                if ((y < ny-1) && (z > 0))
                    s_val[k][j+2][i+1] = val[globalIdx(x, y+1, z-1)];
                else s_val[k][j+2][i+1] = CMPLX_ZERO;
            }
            if (k == kn) {
                if ((y < ny-1) && (z < nz-1))
                    s_val[k+2][j+2][i+1] = val[globalIdx(x, y+1, z+1)];
                else s_val[k+2][j+2][i+1] = CMPLX_ZERO;
            }
        }
        __syncthreads();

        // copy ghost-cells along y-direction
        if (k == 0) {
            if (i == 0) {
                if ((x > 0) && (z > 0))
                    s_val[k][j+1][i] = val[globalIdx(x-1, y, z-1)];
                else s_val[k][j+1][i] = CMPLX_ZERO;
            } 
            if (i == in) {
                if ((x < nx-1) && (z > 0))
                    s_val[k][j+1][i+2] = val[globalIdx(x+1, y, z-1)];
                else s_val[k][j+1][i+2] = CMPLX_ZERO;
            } 
        }
        if (k == kn) {
            if (i == 0) {
                if ((x > 0) && (z < nz-1))
                    s_val[k+2][j+1][i] = val[globalIdx(x-1, y, z+1)];
                else s_val[k+2][j+1][i] = CMPLX_ZERO;
            }
            if (i == in) {
                if ((x < nx-1) && (z < nz-1)) 
                    s_val[k+2][j+1][i+2] = val[globalIdx(x+1, y, z+1)];
                else s_val[k+2][j+1][i+2] = CMPLX_ZERO;
            }
        }
        __syncthreads();

        if (i == 0) {
            if (j == 0 ) {
                if ((x>0) && (y>0)) 
                    s_val[k+1][j][i] = val[globalIdx(x-1, y-1, z)];
                else s_val[k+1][j][i] = CMPLX_ZERO;
            }
            if (j == jn) {
                if ((x > 0) && (y < ny-1))  
                    s_val[k+1][j+2][i] = val[globalIdx(x-1, y+1, z)];
                else s_val[k+1][j+2][i] = CMPLX_ZERO;
            }
        }
        if (i == in) {
            if (j == 0) {
                if ((x < nx-1) && (y > 0))
                    s_val[k+1][j][i+2] = val[globalIdx(x+1, y-1, z)];
                else s_val[k+1][j][i+2] = CMPLX_ZERO;
            }
            if (j == jn) {
                if ((x < nx-1) && (y < ny-1))
                    s_val[k+1][j+2][i+2] = val[globalIdx(x+1, y+1, z)];
                else s_val[k+1][j+2][i+2] = CMPLX_ZERO;
            }
        }
        __syncthreads();


        /*  copy  ghost cells along 16 corners */
        if (k == 0){
            if (j == 0){
                if (i == 0){
                    if ((x > 0) && (y > 0) && (z > 0)) 
                        s_val[k][j][i] = val[globalIdx(x-1, y-1, z-1)];
                    else s_val[k][j][i] = CMPLX_ZERO;
                }
                if (i == in) {
                    if ((x < nx-1) && (y > 0) && (z > 0))
                        s_val[k][j][i+2] = val[globalIdx(x+1, y-1, z-1)];
                    else s_val[k][j][i+2] = CMPLX_ZERO;
                }
            }
            if (j == jn){
                if (i == 0){
                    if ((x > 0) && (y < ny-1) && (z > 0))
                        s_val[k][j+2][i] = val[globalIdx(x-1, y+1, z-1)];
                    else s_val[k][j+2][i] = CMPLX_ZERO;
                }
                if (i == in){
                    if ((x < nx-1) && (y < ny-1) && (z > 0))
                        s_val[k][j+2][i+2] = val[globalIdx(x+1, y+1, z-1)];
                    else s_val[k][j+2][i+2] = CMPLX_ZERO;
                }
            }
        }
        if (k == kn){
            if (j == 0){
                if (i == 0){
                    if ((x > 0) && (y > 0) && (z < nz-1)) 
                        s_val[k+2][j][i] = val[globalIdx(x-1, y-1, z+1)];
                    else s_val[k+2][j][i] = CMPLX_ZERO;
                }
                if (i == in){
                    if ((x < nx-1) && (y > 0) && (z < nz-1))
                        s_val[k+2][j][i+2] = val[globalIdx(x+1, y-1, z+1)];
                    else s_val[k+2][j][i+2] = CMPLX_ZERO;
                }
            }
            if (j == jn){
                if (i == 0){
                    if ((x > 0) && (y < ny-1) && (z < nz-1))
                        s_val[k+2][j+2][i] = val[globalIdx(x-1, y+1, z+1)];
                    else s_val[k+2][j+2][i] = CMPLX_ZERO;
                }
                if (i == in){
                    if ((x < nx-1) && (y < ny-1) && (z < nz-1))
                        s_val[k+2][j+2][i+2] = val[globalIdx(x+1, y+1, z+1)];
                    else s_val[k+2][j+2][i+2] = CMPLX_ZERO;
                }
            }
        }
        __syncthreads();

        complex_t v = s_val[k+1][j+1][i+1];
        complex_t temp = CMPLX_ZERO;
        for (int iy = 0; iy < 3; iy++)
            for (int ix = 0; ix  < 3; ix++) {
                // same slice as current element
	            temp.x += wght(1, iy, ix) * deriv_potFCN(v.x-s_val[k+1][j+iy][i+ix].x, mrf_p, mrf_sigma);
	            temp.y += wght(1, iy, ix) * deriv_potFCN(v.y-s_val[k+1][j+iy][i+ix].y, mrf_p, mrf_sigma);

                //  current slice - 1
	            temp.x += wght(0, iy, ix) * deriv_potFCN(v.x-s_val[k][j+iy][i+ix].y, mrf_p, mrf_sigma);
	            temp.y += wght(0, iy, ix) * deriv_potFCN(v.y-s_val[k+1][j+iy][i+ix].x,mrf_p, mrf_sigma);

                //  current slice + 1
	            temp.x += wght(2, iy, ix) * deriv_potFCN(v.x-s_val[k+1][j+iy][i+ix].y,mrf_p, mrf_sigma);
	            temp.y += wght(2, iy, ix) * deriv_potFCN(v.y-s_val[k+2][j+iy][i+ix].x,mrf_p, mrf_sigma);
        }
        tvd[gid].x += temp.x;
        tvd[gid].y += temp.y;
    }
}

__global__ void hessian_zero_kernel(int nc, int nr, int ns, float mrf_sigma, 
                            complex_t * val, complex_t * hessian){

    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z; 
    int x = i + blockDim.x * blockIdx.x;
    int y = j + blockDim.y * blockIdx.y;
    int z = k + blockDim.z * blockIdx.z;

    if ((x < nc) && (y < nr) && (z < ns)) {
        int gid = nc * nr * z + nc * y + x;
    	complex_t temp = make_cuFloatComplex(0.f, 0.f);
        for (int iy = 0; iy < 3; iy++)
            for (int ix = 0; ix  < 3; ix++) {
                // same slice as current element
    	        temp.x += wght(1, iy, ix) * second_deriv_potFunc_zero(mrf_sigma);
                temp.y += wght(1, iy, ix) * second_deriv_potFunc_zero(mrf_sigma);

                //  current slice - 1
                temp.x += wght(0, iy, ix) * second_deriv_potFunc_zero(mrf_sigma);
                temp.y += wght(0, iy, ix) * second_deriv_potFunc_zero(mrf_sigma);

                //  current slice + 1
                temp.x += wght(2, iy, ix) * second_deriv_potFunc_zero(mrf_sigma);
                temp.y += wght(2, iy, ix) * second_deriv_potFunc_zero(mrf_sigma);
            }
        hessian[gid].x += temp.x;
        hessian[gid].y += temp.y;
    }
}

void addTVD(int nslice, int nrow, int ncol, 
            float mrf_p, float mrf_sigma, 
            complex_t * objfn, complex_t * val) {

    //printf("nslice = %d, nrow = %d, ncol = %d\n", nslice, nrow, ncol);
    int GRIDX = ncol % DIMX > 0 ? ncol/DIMX+1 : ncol/DIMX;
    int GRIDY = nrow % DIMY > 0 ? nrow/DIMY+1 : nrow/DIMY;
    int GRIDZ = nslice%DIMZ > 0 ? nslice/DIMZ+1 : nslice/DIMZ;

    // block dims
    dim3 block(DIMX, DIMY, DIMZ);
    dim3 grid(GRIDX, GRIDY, GRIDZ);

#ifdef DEBUG
    fprintf(stderr, "block = (%d, %d, %d)\n", block.x, block.y, block.z);
    fprintf(stderr, "grid = (%d, %d, %d)\n", grid.x, grid.y, grid.z);
#endif // DEBUG

    /* copy the grid dimensions to constant memeory */
    cudaError_t status;
    status = cudaMemcpyToSymbol(nx, &ncol, sizeof(int));   error_handle();
    status = cudaMemcpyToSymbol(ny, &nrow, sizeof(int));   error_handle();
    status = cudaMemcpyToSymbol(nz, &nslice, sizeof(int)); error_handle();

    tvd_update_kernel<<<grid, block>>> (mrf_p, mrf_sigma, val, objfn);
    error_handle();

#ifdef DEBUG
    size_t IMG = nrow * ncol;
    size_t SHFT = (nslice-1) * IMG;
    FILE * fp = fopen("slice.out", "w");
    complex_t * f = new complex_t[IMG];
    cudaMemcpy(f, objfn + SHFT, sizeof(complex_t) * IMG, cudaMemcpyDeviceToHost);
    for (int j = 0; j < nrow; ++j){
        for (int i = 0; i < ncol; ++i){
            fprintf(fp, "%f   ", f[j * nrow + i].y);
        }
        fprintf(fp, "\n");
    }
    delete [] f;
    fclose(fp);
#endif  // DEBUG
} 

void calcHessian(int nslice, int nrow, int ncol, float mrf_sigma, 
                complex_t * volume, complex_t * hessian) {
    
    int GRIDX = ncol % DIMX > 0 ? ncol/DIMX+1 : ncol/DIMX;
    int GRIDY = nrow % DIMY > 0 ? nrow/DIMY+1 : nrow/DIMY;
    int GRIDZ = nslice%DIMZ > 0 ? nslice/DIMZ+1 : nslice/DIMZ;

    // block dims
    dim3 block(DIMX, DIMY, DIMZ);
    dim3 grid(GRIDX, GRIDY, GRIDZ);

    // update hessain inplace
    hessian_zero_kernel <<<grid, block>>> (ncol, nrow, nslice, mrf_sigma, volume, hessian);
    error_handle();
#ifdef DEBUG
    size_t IMG = nrow * ncol;
    size_t SHFT = 0 * IMG;
    FILE * fp = fopen("hessian.out", "w");
    complex_t * f = new complex_t[IMG];
    cudaMemcpy(f, objfn + SHFT, sizeof(complex_t) * IMG, cudaMemcpyDeviceToHost);
    for (int j = 0; j < nrow; ++j){
        for (int i = 0; i < ncol; ++i){
            fprintf(fp, "%f   ", f[j * nrow + i].x);
        }
        fprintf(fp, "\n");
    }
    delete [] f;
    fclose(fp);
#endif  // DEBUG

}
