#include <iostream>

#include <cuda.h>
#include <cufinufft.h>

#include "common.h"
#include "utils.cuh"

#ifndef NUFFT__H
#define NUFFT__H

#define NUFFT_CALL(ans) { nufft_check((ans), __FILE__, __LINE__); }
inline void nufft_check(int code, const char *file, int line) {
    if (code != 0) {
        std::cerr << "nufft error at: " << file << ":" << line << std::endl;
        exit(code);
    }
}

namespace tomocam {

    struct NUFFTGrid {
        int M;
        int gpu_device_id;
        float *x;
        float *y;

        NUFFTGrid(int nc, int np, float *angles, float center, int gid) {
            M = nc * np;
            gpu_device_id = gid;
            size_t bytes = nc * np * sizeof(float);
            SAFE_CALL(cudaMalloc(&x, bytes));
            SAFE_CALL(cudaMalloc(&y, bytes));
            nufft_grid(nc, np, x, y, angles, center);
        }

        ~NUFFTGrid() {
            SAFE_CALL(cudaFree(x));
            SAFE_CALL(cudaFree(y));
        }
    };

    inline int nufft_setgrid(NUFFTGrid &grid, cufinufftf_plan &plan) {
        int ier = cufinufftf_setpts(grid.M, grid.x, grid.y, NULL, 0, NULL, NULL, NULL, plan);
        return ier;
    }
   
    inline int nufftPlan1(dim3_t dims, NUFFTGrid &grid, cufinufftf_plan &plan) {
        int type = 1;
        int ndim = 2;
        int nmodes[] = {dims.z, dims.z, 1};
        int iflag = 1;
        int ntransf = dims.x;
        float tol = 1.e-06;

        int err;
        // multi-gpu
        cufinufft_opts opts;
        err = cufinufft_default_opts(type, ndim, &opts);
        if (err != 0) return err;

        opts.gpu_device_id = grid.gpu_device_id;
        err = cufinufftf_makeplan(type, ndim, nmodes, iflag, ntransf, tol, 0, &plan, &opts);
        if (err != 0) return err;

        err = nufft_setgrid(grid, plan);
        return err;
    }

    inline int nufftPlan2(dim3_t dims, NUFFTGrid &grid, cufinufftf_plan &plan) {
        int type = 2;
        int ndim = 2;
        int nmodes[] = {dims.z, dims.z, 1};
        int iflag = -1;
        int ntransf = dims.x;
        float tol = 1.e-06;

        int err;
        // multi-gpu
        cufinufft_opts opts;
        err = cufinufft_default_opts(type, ndim, &opts);
        if (err != 0) return err;
        // set device
        opts.gpu_device_id = grid.gpu_device_id; 
        err = cufinufftf_makeplan(type, ndim, nmodes, iflag, ntransf, tol, 0, &plan, &opts);

        // get grid points
        err = nufft_setgrid(grid, plan);
        if (err != 0) return err;
        return err;
    }
}

#endif // NUFFT__H
