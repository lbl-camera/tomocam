#include <stdlib.h>
#include <stdio.h>

#include <arrayfire.h>
#include "polarsample.h"
#include <cuda_profiler_api.h>

int main(int argc, char ** argv){

    if (argc != 3){
        printf("Usage: %s <num of lines> <num of angles>\n", argv[0]);
        exit(1);
    }

    int num_lines = atoi(argv[1]);
    int num_angs = atoi(argv[2]);

    // generate polargrid
    size_t npts = num_angs * num_lines;
    int cen = num_lines / 2;
    float d_th = af::Pi / num_angs;
    af::array pts;
    { 
        af::array ra = af::iota(af::dim4(num_lines));
        af::array th = af::transpose(af::iota(af::dim4(num_angs))) * d_th;

        af::array x = af::matmul(ra-cen, af::cos(th)) + cen;
        af::array y = af::matmul(ra-cen, af::sin(th)) + cen;
        pts = af::complex(x, y);
    }

    af::array sval = af::complex(af::randu(num_lines, num_angs), 
									af::randu(num_lines, num_angs));
   
    size_t ngrid = num_lines * num_lines;
    uint2 grid_size = { num_lines, num_lines };

    // ouytput array
	af::array gval(num_lines, num_lines, c32);

    int kb_table_size = 128;
    float kb_table_scale = 1.7f;
    float kernel_radius = 3.f;
    af::array kblut = af::iota(af::dim4(kb_table_size));
    kblut = af::exp(-0.5 * kblut * kblut / 400.);
    
    complex_t * pp = (complex_t *) pts.device<void>();
    complex_t * sv = (complex_t *) sval.device<void>();
    complex_t * gv = (complex_t *) gval.device<void>();
    float * kb = kblut.device<float>();
    af::sync();
    af::timer start1 = af::timer::start();

    cudaProfilerStart();
    polarsample_transpose(pp, sv, npts, grid_size, kb,
            kb_table_size, kb_table_scale, kernel_radius, gv);
    cudaProfilerStop();
    printf("elapsed seconds: %g\n", af::timer::stop(start1));

	gval.unlock();
    return 0;
}
