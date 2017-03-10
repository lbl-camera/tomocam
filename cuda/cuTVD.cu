
#include "polarsample.h"

const int WORK = 8;

__constant__ int nx, ny, nz;

__inline__ __device__ complex_t filter(complex_t v) {
    return make_cuFloatComplex(fabs(v.x), fabs(f.y));
}

__inline__ __device__ int globalIdx(int i, int j, int k){
    return (nx * ny * k + nx * j + i);
}

__global__ void tvd_update_kernel(complex_t * val, complex_t *grad){
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z; 
    int xOffset = blockDim.x * blockIdx.x;
    int yOffset = blockDim.y * blockIdx.y;
    int zOffset = blockDim.z * blockIdx.z;
    int x = i + xOffset;
    int y = j + yOffset;
    int z = k + zOffset;
       
    if ((x < nx) && (y < ny) && (z < nz)) {

        /* 
         * e.g. sNX = 4, sNY = 4, sNZ = 128
         */
        const int sNX = blockDim.x + 2;
        const int sNY = blockDim.y + 2;
        const int sNZ = blockDim.z + 2;
        int gid = z * nx * ny + y * nx + z;

        /* copy values into shared memory. 
         * Max size of shared memory = 64 x 1024 Bytes
         * which translates to  8192 complex number
         */

        const CMPLX_ZERO = make_cuFloatComplex(0.f, 0.f);
        __shared__ complex_t s_val[sNX][sNY][sNZ];

        // copy from global memory
        s_val[i+1][j+1][k+1] = val[gid];

        /* copy ghost cells, except corners */
        if (i == 0){
            if (x > 0) s_val[i][j][k] = val[globalIdx(x-1, y, z)];
            else s_val[i][j][k] = CMPLX_ZERO;
        }

        if (j == 0){
            if (y > 0) s_val[i][j][k] = val[globalIdx(x, y-1, z)];
            else s_val[i][j][k] = CMPLX_ZERO;
        }

        if (k == 0){
            if (z > 0) s_val[i][j][k] = val[globalIdx(x, y, z-1)];
            else s_val[i][j][k] = CMPLX_ZERO;
        }

        int xlen = min(sNX, nx - xOffset);
        if (i == xlen-1) {
            if (xOffset + xlen < nx) s_val[i+2][j][k] = val[gid+1];
            else s_val[i+2][j][k] = CMPLX_ZERO;
        }

        int ylen = min(sNY, ny - yOffset);
        if (j == ylen-1) {
            if (yOffset + ylen < ny) s_val[i][j+2][k] = val[globalIdx(x, y+1,z)];
            else s_val[i][j+2][k] = CMPLX_ZERO;
        }

        int zlen = min(sNZ, nz - zOffset);
        if (k == zlen-1) {
            if (zOffset + zlen < nz) s_val[i][j][k+2] = val[globalIdx(x, y, z+1)];
            else s_val[i][j][k+2] = CMPLX_ZERO;
        }

        __synchthreads();

        /* copy the corners, all eight of them */
        if (k == 0){
            if (j == 0){
                if (i == 0){
                    if ((x > 0) && (y > 0) && (z > 0)) 
                        s_val[i][j][k] = val[globalIdx(x-1,y-1,z-1)];
                    else s_val[i][j][k] = CMPLX_ZERO;
                }
                if (i == xlen-1) {
                    if (xOffset + xlen < nx)
                        s_val[i+2][j][k] = val[globalIdx(x+1, y-1, z-1)];
                    else s_val[i+2][j][k] = CMPLX_ZERO;
                }
            }
            if (j == ylen-1){
                if (i == 0){
                    if ((x > 0) && (yOffset + ylen < ny) && (z > 0))
                        s_val[i][j+2][k] = val[globalIdx(x-1, y+1, z-1)];
                    else s_val[i][j+2][k] = CMPLX_ZERO;
                }
                if (i == xlen-1){
                    if ((xOffset + xlen < nx) && (yOffset + ylen < ny) && (z > 0))
                        s_val[i+2][j+2][k] = val[globalIdx(x+1, y+1, z-1)];
                    else s_val[i+2][j+2][k] = CMPLX_ZERO;
                }
            }
        }
        if (k == zlen-1){
            if (j == 0){
                if (i == 0){
                    if ((x > 0) && (y > 0) && (zOffset + zlen < nz)) 
                        s_val[i][j][k+2] = val[globalIdx(x-1,y-1,z+1)];
                    else s_val[i][j][k+2] = CMPLX_ZERO;
                }
                if (i == xlen-1){
                    if (xOffset + xlen < nx)
                        s_val[i+2][j][k+2] = val[globalIdx(x+1, y-1, z+1)];
                    else s_val[i+2][j][k+2] = CMPLX_ZERO;
                }
            }
            if (j == ylen-1){
                if (i == 0){
                    if ((x > 0) && (yOffset + ylen < ny) && (zOffset + zlen < nz))
                        s_val[i][j+2][k+2] = val[globalIdx(x-1, y+1, z+1)];
                    else s_val[i][j+2][k+2] = CMPLX_ZERO;
                }
                if (i == xlen-1){
                    if ((xOffset + xlen < nx) && (yOffset + ylen < ny) && (zOffset + zlen < nz))
                        s_val[i+2][j+2][k+2] = val[globalIdx(x+1, y+1, z+1)];
                    else s_val[i+2][j+2][k+2] = CMPLX_ZERO;
                }
            }
        }
        __synchthreads();
        complex_t v = s_val[i+1][j+1][k+1];
        for (int ix = 0; ix  < 3; ix++)
            for (int iy = 0; iy < 3; iy++) 
                for (int iz = 0; iz < 3; iz++) 
                    grad[gid] += filter(s_val[i+ix][j+iy][k+iz] - v);

    }
}


void addTVD(int nrow, int ncol, int nslice, complex_t * objfn, complex_t * val) {

    const DIMX = 16;
    const DIMY = 16;
    const DIMZ = 2;
    dim3 blocks(DIMX, DIMY, DIMZ);

    int GRIDX = (nx / DIMX) + 1;
    int GRIDY = (ny / DIMY) + 1;
    int GRIDZ = (nz / DIMZ) + 1;
    dim3 grid(GRIDX, GRIDY, GRIDZ);

    cudaError_t status;
    status = cuMemcpyToSymbol(nx, &ncol, 1);   error_handle();
    status = cuMemcpyToSymbol(ny, &nrow, 1);   error_handle();
    status = cuMemcpyToSymbol(nz, &nslice, 1); error_handle();
    tvd_update_kernel<<<grid, blocks>>> (val, objfn);
    error_handle();
}
