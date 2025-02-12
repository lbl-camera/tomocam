#include <fftw3.h>
#include <omp.h>

#include <iostream>
#include <cmath>
#include <ctime>

#include <cuda_profiler_api.h>

#include "dist_array.h"
#include "fft.h"


void fft(fftwf_complex * input, fftwf_complex * output, tomocam::dim3_t dims) {
    int rank = 1;
    int n[] = { dims.z };
    int idist = dims.z;
    int odist = dims.z;
    int istride = 1;
    int ostride = 1;
    int *inembed = NULL;
    int *onembed = NULL;
    int batches = dims.x * dims.y;
    fftwf_plan plan = fftwf_plan_many_dft(rank, n, batches, input, inembed,
        istride, idist, output, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

void ifft(fftwf_complex * input, fftwf_complex * output, tomocam::dim3_t dims) {
    int rank = 1;
    int n[] = { dims.z };
    int idist = dims.z;
    int odist = dims.z;
    int istride = 1;
    int ostride = 1;
    int *inembed = NULL;
    int *onembed = NULL;
    int batches = dims.x * dims.y;
    fftwf_plan plan = fftwf_plan_many_dft(rank, n, batches, input, inembed,
        istride, idist, output, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

int main() {
    std::srand(100);

    // create a random array
    tomocam::dim3_t dims(211, 128, 128);
    tomocam::DArray<complex_t> inArr(dims);

    complex_t * input = inArr.data();
    for (int i = 0; i < inArr.size(); i++) {
        float x = (float) i;
        float y = 0.f;
        input[i] = complex_t(x, y);
    }


    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    tomocam::DArray<complex_t> outArr(dims);
    tomocam::DArray<complex_t> outArr2(dims);
    complex_t * output = outArr.data();
    tomocam::dim3_t d = inArr.dims();

    clock_t ts0 = std::clock();
    fft(reinterpret_cast<fftwf_complex *>(input), reinterpret_cast<fftwf_complex *>(output), d);
    clock_t ts1 = std::clock();
    //cudaProfilerStart();
    tomocam::fft1d(inArr, outArr2);
    //cudaProfilerStop();
    clock_t ts2 = std::clock();

    std::cout << "fftw = " << (ts1 - ts0) << std::endl;
    std::cout << "cufft = " << (ts2 - ts1) << std::endl;

    for (int i = 0; i < 3; i++ ){
        for (int j = 0; j < 10; j++)
            std::cout << outArr(10*i, 0, j) << ",      " << outArr2(10*i, 0, j) << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
     

