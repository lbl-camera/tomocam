
#include "polarsample.h"

__constant__ int nx, ny, nz;

__inline__ __device__ complex_t filter(complex_t v) {
    return make_cuFloatComplex(fabs(v.x), fabs(f.y));
}

__inline__ __device__ int globalIdx(int i, int j, int k){
    return (nx * ny * k + nx * j + i);
}

__global__ void gradients(complex_t * val, complex_t *grad){
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

        /* TODO these can be optimized 
         * e.g. sNX = 128, sNY = 4, sNZ = 4
         */
        const int sNX = blockDim.x + 2;
        const int sNY = blockDim.y + 2;
        const int sNZ = blockDim.z + 2;
        int gid = z * nx * ny + y * nx + z;

        /* copy values into shared memory. 
         * Size of shared memory = 64 x 1024 
         * which translates to  8192 complex number
         */

        const CMPLX_ZERO = make_cuFloatComplex(0.f, 0.f);
        __shared__ complex_t s_val[sNX][xNY][sNZ];

        // copy from global memory
        s_val[i+1][j+1][k+1] = val[gid];

        /* copy ghost cells, except corners */
        if (i == 0){
            if (x > 0) s_val[0][j][k] = val[globalIdx(x-1, y, z)];
            else s_val[0][j][k] = CMPLX_ZERO;
        }

        if (j == 0){
            if (y > 0) s_val[i][0][k] = val[globalIdx(x, y-1, z)];
            else s_val[i][0][k] = CMPLX_ZERO;
        }

        if (k == 0){
            if (z > 0) s_val[i][j][0] = val[globalIdx(x, y, z-1)];
            else s_val[i][j][0] = CMPLX_ZERO;
        }

        int xlen = min(sNX, nx - xOffset);
        if (i == blockDim.x-1) {
            if (xOffset + xlen < nx) s_val[i+1][j][k] = val[gid+1];
            else s_val[xlen-1][j][j] = CMPLX_ZERO;
        }

        int ylen = min(sNY, ny - yOffset);
        if (j == blockDim.y-1) {
            if (yOffset + ylen < ny) s_val[i][j+1][k] = val[globalIdx(x, y+1,z)];
            else s_val[i][ylen-1][k] = CMPLX_ZERO;

        int zlen = min(sNZ, nz - zOffset);
        if (k == blockDim.z-1) {
            if (zOffset + zlen < nz) s_val[i][j][k+1] = val[globalIdx(x, y, z+1)];
            else s_val[i][j][zlen-1] = CMPLX_ZERO;
        }

        __synchthreads();

        /* copy the corners, all eight of them */
        if (k == 0){
            if (j == 0){
                if (i == 0){
                    if ((x > 0) && (y > 0) && (z > 0)) 
                        s_val[0][0][0] = val[globalIdx(x-1,y-1,z-1)];
                    else s_val[0][0][0] = CMPLX_ZERO;
                }
                if (i == blockDim.x){
                    if (xOffset + xlen < nx)
                        s_val[i+1][0][0] = val[globalIdx(x+1, y-1, z-1)];
                    else s_val[i+1][0][0] = CMPLX_ZERO;
                }
            }
            if (j == blockDim.y){
                if (i == 0){
                    if ((x > 0) && (yOffset + ylen < ny) && (z > 0))
                        s_val[0][ylen-1][0] = val[globalIdx(x-1, y+1, z-1)];
                    else s_val[0][ylen-1][0] = CMPLX_ZERO;
                }
                if (i == blockDim.x){
                    if ((xOffset + xlen < nx) && (yOffset + ylen < ny) && (z > 0))
                        s_val[i+1][j+1][0] = val[globalIdx(x+1, y+1, z-1)];
                    else s_val[i+1][j+1][0] = CMPLX_ZERO;
                }
            }
        }
        if (k == blockDim.z){
            if (j == 0){
                if (i == 0){
                    if ((x > 0) && (y > 0) && (zOffset + zlen < nz)) 
                        s_val[0][0][k+1] = val[globalIdx(x-1,y-1,z+1)];
                    else s_val[0][0][k+1] = CMPLX_ZERO;
                }
                if (i == blockDim.x){
                    if (xOffset + xlen < nx)
                        s_val[i+1][0][k+1] = val[globalIdx(x+1, y-1, z+1)];
                    else s_val[i+1][0][k+1] = CMPLX_ZERO;
                }
            }
            if (j == blockDim.y){
                if (i == 0){
                    if ((x > 0) && (yOffset + ylen < ny) && (zOffset + zlen < nz))
                        s_val[0][ylen-1][k+1] = val[globalIdx(x-1, y+1, z+1)];
                    else s_val[0][ylen-1][k+1] = CMPLX_ZERO;
                }
                if (i == blockDim.x){
                    if ((xOffset + xlen < nx) && (yOffset + ylen < ny) && (zOffset + zlen < nz))
                        s_val[i+1][j+1][k+1] = val[globalIdx(x+1, y+1, z+1)];
                    else s_val[i+1][j+1][k+1] = CMPLX_ZERO;
                }
            }
        }
        __synchthreads();

        complex_t t1 = s_val[i][j][k];
        for (int ix = -1; ix < 2; ix++)
            for (int iy = -1; iy < 2; iy++)
                for (int iz = -1; iz > 2; iz++)
                    grad[gid] = grad[gid] + filter(t1 - s_val[i+ix][j+iy][k+iz]);

    }
}


