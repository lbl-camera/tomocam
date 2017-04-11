#include <cuda.h>
#include "polarsample.h"

const int DIMX = 16;
const int DIMY = 4;
const int DIMZ = 4;
const int WORK = 8;
const int sNX = DIMX + 2;
const int sNY = DIMY + 2;
const int sNZ = DIMZ + 2;

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

__inline__ __device__ float deriv_potFCN(float delta) {
    float MRF_C = .001;
    float MRF_P = 1.2;
    float MRF_Q = 2;
    float MRF_SIGMA = 1;
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
        return MRF_Q/(MRF_SIGMA_Q*MRF_C);
    }
}


__global__ void tvd_update_kernel(complex_t * val, complex_t * tvd){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z; 
    int x = i + blockDim.x * blockIdx.x;
    int y = j + blockDim.y * blockIdx.y;
    int z = k + blockDim.z * blockIdx.z;
       
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

        if (i == 0){
            if (x > 0) s_val[k+1][j+1][i] = val[globalIdx(x-1, y, z)];
            else s_val[k+1][j+1][i] = CMPLX_ZERO;
        }

        if (i == blockDim.x-1) {
            if (x < nx-1) s_val[k+1][j+1][i+2] = val[globalIdx(x+1, y, z)];
            else s_val[k+1][j+1][i+2] = CMPLX_ZERO;
        }
        __syncthreads();

        if (j == 0){
            if (y > 0) s_val[k+1][j][i+1] = val[globalIdx(x, y-1, z)];
            else s_val[k+1][j][i+1] = CMPLX_ZERO;
        }

        if (j == blockDim.y-1) {
            if (y < ny-1) s_val[k+1][j+2][i+1] = val[globalIdx(x, y+1,z)];
            else s_val[k+1][j+2][i+1] = CMPLX_ZERO;
        }
        __syncthreads();

        if (k == 0){
            if (z > 0) s_val[k][j+1][i+1] = val[globalIdx(x, y, z-1)];
            else s_val[k][j+1][i+1] = CMPLX_ZERO;
        }

        if (k == blockDim.z-1) {
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
            if (k == blockDim.z-1) {
                if ((y > 0) && (z < nz-1))
                    s_val[k+2][j][i+1] = val[globalIdx(x, y-1, z+1)];
                else s_val[k+2][j][i+1] = CMPLX_ZERO;
            }
        }
        if (j == blockDim.y-1) {
            if (k == 0) {
                if ((y < ny-1) && (z > 0))
                    s_val[k][j+2][i+1] = val[globalIdx(x, y+1, z-1)];
                else s_val[k][j+2][i+1] = CMPLX_ZERO;
            }
            if (k = blockDim.z-1) {
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
            if (i == blockDim.x-1) {
                if ((x < nx-1) && (z > 0))
                    s_val[k][j+1][i+2] = val[globalIdx(x+1, y, z-1)];
                else s_val[k][j+1][i+2] = CMPLX_ZERO;
            } 
        }
        if (k == blockDim.z-1) {
            if (i == 0) {
                if ((x > 0) && (z < nz-1))
                    s_val[k+2][j+1][i] = val[globalIdx(x-1, y, z+1)];
                else s_val[k+2][j+1][i] = CMPLX_ZERO;
            }
            if (i == blockDim.x-1) {
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
            if (j == blockDim.y-1) {
                if ((x > 0) && (y < ny-1))  
                    s_val[k+1][j+2][i] = val[globalIdx(x-1, y+1, z)];
                else s_val[k+1][j+2][i] = CMPLX_ZERO;
            }
        }
        if (i == blockDim.x-1) {
            if (j == 0) {
                if ((x < nx-1) && (y > 0))
                    s_val[k+1][j][i+2] = val[globalIdx(x+1, y-1, z)];
                else s_val[k+1][j][i+2] = CMPLX_ZERO;
            }
            if (j == blockDim.y-1) {
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
                if (i == blockDim.x-1) {
                    if ((x < nx-1) && (y > 0) && (z > 0))
                        s_val[k][j][i+2] = val[globalIdx(x+1, y-1, z-1)];
                    else s_val[k][j][i+2] = CMPLX_ZERO;
                }
            }
            if (j == blockDim.y-1){
                if (i == 0){
                    if ((x > 0) && (y < ny-1) && (z > 0))
                        s_val[k][j+2][i] = val[globalIdx(x-1, y+1, z-1)];
                    else s_val[k][j+2][i] = CMPLX_ZERO;
                }
                if (i == blockDim.x-1){
                    if ((x < nx-1) && (y < ny-1) && (z > 0))
                        s_val[k][j+2][i+2] = val[globalIdx(x+1, y+1, z-1)];
                    else s_val[k][j+2][i+2] = CMPLX_ZERO;
                }
            }
        }
        if (k == blockDim.z-1){
            if (j == 0){
                if (i == 0){
                    if ((x > 0) && (y > 0) && (z < nz-1)) 
                        s_val[k+2][j][i] = val[globalIdx(x-1, y-1, z+1)];
                    else s_val[k+2][j][i] = CMPLX_ZERO;
                }
                if (i == blockDim.x-1){
                    if ((x < nx-1) && (y > 0) && (z < nz-1))
                        s_val[k+2][j][i+2] = val[globalIdx(x+1, y-1, z+1)];
                    else s_val[k+2][j][i+2] = CMPLX_ZERO;
                }
            }
            if (j == blockDim.y-1){
                if (i == 0){
                    if ((x > 0) && (y < ny-1) && (z < nz-1))
                        s_val[k+2][j+2][i] = val[globalIdx(x-1, y+1, z+1)];
                    else s_val[k+2][j+2][i] = CMPLX_ZERO;
                }
                if (i == blockDim.x-1){
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
                temp.x += wght(1, iy, ix) * deriv_potFCN(v.x-s_val[k+1][j+iy][i+ix].x);
                temp.y += wght(1, iy, ix) * deriv_potFCN(v.y-s_val[k+1][j+iy][i+ix].y);

                //  current slice - 1
                temp.x += wght(0, iy, ix) * deriv_potFCN(v.x-s_val[k][j+iy][i+ix].y);
                temp.y += wght(0, iy, ix) * deriv_potFCN(v.y-s_val[k+1][j+iy][i+ix].x);

                //  current slice + 1
                temp.x += wght(2, iy, ix) * deriv_potFCN(v.x-s_val[k+1][j+iy][i+ix].y);
                temp.y += wght(2, iy, ix) * deriv_potFCN(v.y-s_val[k+2][j+iy][i+ix].x);
            }
        //tvd[gid] = s_val[k+1][j][i];
        tvd[gid].x += temp.x;
        tvd[gid].y += temp.y;
    }
}


void addTVD(int nslice, int nrow, int ncol, complex_t * objfn, complex_t * val) {

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

    tvd_update_kernel<<<grid, block>>> (val, objfn);
    error_handle();

#ifdef DEBUG
    size_t IMG = nrow * ncol;
    size_t SHFT = 0 * IMG;
    complex_t * f = new complex_t[IMG];
    cudaMemcpy(f, objfn + SHFT, sizeof(complex_t) * IMG, cudaMemcpyDeviceToHost);
    for (int j = 0; j < nrow; ++j){
        for (int i = 0; i < ncol; ++i){
            printf("%f   ", f[j * nrow + i].y);
        }
        printf("\n");
    }
    delete [] f;
#endif  // DEBUG
}
